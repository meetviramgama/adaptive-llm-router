"""
Cloud Pipeline
--------------
Same logic as src/router/pipeline.py but uses Groq API
instead of local Ollama for cloud deployment.

Components:
    Classifier : llama-3.1-8b-instant  (fast, cheap)
    Compressor : llama-3.1-8b-instant  (fast, cheap)
    Simple ans : llama-3.1-8b-instant  (fast, cheap)
    Complex ans: llama-3.3-70b-versatile (capable)
    Cache      : FAISS + nomic-embed-text via Groq embeddings
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

from groq_client import generate, LOW_MODEL, HIGH_MODEL


# ============================================================================
# Classifier prompt
# ============================================================================

CLASSIFIER_PROMPT = """You are a binary classifier. Read the query and reply with exactly one word.

Reply COMPLEX if the query requires explanation, analysis, comparison, reasoning, or multiple steps.
Reply SIMPLE if the query needs only a short factual answer.

Reply with ONE word only: SIMPLE or COMPLEX"""

# ============================================================================
# Compression prompt
# ============================================================================

COMPRESS_SYSTEM = """You are a query compression assistant.
Rewrite the complex user query into a precise structured brief.

Output ONLY these three things:
1. INTENT: What exactly is being asked (one sentence)
2. CONSTRAINTS: Key requirements or context (one sentence)
3. OUTPUT: What format or depth of answer is expected (one sentence)

Maximum 3 sentences. No padding. Do not answer the query."""


# ============================================================================
# Response dataclass
# ============================================================================

@dataclass
class PipelineResponse:
    query:            str
    answer:           str
    complexity:       str
    cache_hit:        bool
    model_used:       str
    classifier_s:     float
    compress_s:       float
    generation_s:     float
    total_s:          float
    input_tokens:     int
    output_tokens:    int
    compressed_query: str


# ============================================================================
# Semantic cache (FAISS)
# ============================================================================

class SimpleCache:
    """
    Lightweight in-memory semantic cache using Groq embeddings.
    For HuggingFace Spaces — no local Ollama needed.
    """
    def __init__(self, threshold: float = 0.85):
        self.threshold = threshold
        self.entries   = []   # list of (query, answer, embedding)
        self._ready    = False
        self._init()

    def _init(self):
        try:
            import faiss
            import numpy as np
            self._faiss  = faiss
            self._np     = np
            self._index  = faiss.IndexFlatIP(1536)  # nomic-embed dimension
            self._ready  = True
        except Exception:
            self._ready = False

    def _embed(self, text: str):
        try:
            from groq import Groq
            client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
            resp = client.embeddings.create(
                model = "nomic-embed-text-v1.5",
                input = text,
            )
            vec = self._np.array([resp.data[0].embedding], dtype=self._np.float32)
            self._faiss.normalize_L2(vec)
            return vec
        except Exception:
            return None

    def lookup(self, query: str):
        if not self._ready or self._index.ntotal == 0:
            return None
        vec = self._embed(query)
        if vec is None:
            return None
        scores, indices = self._index.search(vec, k=1)
        if float(scores[0][0]) >= self.threshold:
            return self.entries[int(indices[0][0])][1]
        return None

    def store(self, query: str, answer: str):
        if not self._ready:
            return
        vec = self._embed(query)
        if vec is None:
            return
        self._index.add(vec)
        self.entries.append((query, answer))

    def clear(self):
        import faiss
        self._index  = faiss.IndexFlatIP(1536)
        self.entries = []

    def stats(self):
        return {"total_entries": len(self.entries), "threshold": self.threshold}


# ============================================================================
# Cloud Pipeline
# ============================================================================

class Pipeline:
    def __init__(self, use_cache: bool = True, use_compression: bool = True):
        self.use_cache       = use_cache
        self.use_compression = use_compression
        self.cache           = SimpleCache() if use_cache else None

    def _classify(self, query: str) -> tuple[str, float]:
        response = generate(
            model       = LOW_MODEL,
            prompt      = query,
            system      = CLASSIFIER_PROMPT,
            max_tokens  = 10,
            temperature = 0.0,
        )
        raw   = response.text.strip().upper()
        label = "COMPLEX" if "COMPLEX" in raw else "SIMPLE"
        return label, response.wall_clock_s

    def _compress(self, query: str) -> tuple[str, float]:
        response = generate(
            model       = LOW_MODEL,
            prompt      = query,
            system      = COMPRESS_SYSTEM,
            max_tokens  = 200,
            temperature = 0.3,
        )
        return response.text, response.wall_clock_s

    def answer(self, query: str) -> PipelineResponse:
        start = time.time()

        # Step 1: Cache lookup
        if self.use_cache and self.cache:
            cached = self.cache.lookup(query)
            if cached:
                return PipelineResponse(
                    query            = query,
                    answer           = cached,
                    complexity       = "CACHED",
                    cache_hit        = True,
                    model_used       = "cache",
                    classifier_s     = 0.0,
                    compress_s       = 0.0,
                    generation_s     = 0.0,
                    total_s          = round(time.time() - start, 3),
                    input_tokens     = 0,
                    output_tokens    = 0,
                    compressed_query = "",
                )

        # Step 2: Classify
        complexity, classifier_s = self._classify(query)

        # Step 3: Route
        gen_start        = time.time()
        compressed_query = ""
        compress_s       = 0.0

        if complexity == "SIMPLE":
            response = generate(
                model       = LOW_MODEL,
                prompt      = query,
                system      = "You are a helpful assistant. Answer clearly and concisely.",
                max_tokens  = 500,
                temperature = 0.7,
            )
        else:
            if self.use_compression:
                compressed_query, compress_s = self._compress(query)
                prompt_for_high = compressed_query
            else:
                prompt_for_high = query

            response = generate(
                model       = HIGH_MODEL,
                prompt      = prompt_for_high,
                system      = "You are a helpful assistant. Answer thoroughly and accurately.",
                max_tokens  = 2000,
                temperature = 0.7,
            )

        generation_s = time.time() - gen_start
        total_s      = time.time() - start

        # Step 4: Cache
        if self.use_cache and self.cache:
            self.cache.store(query, response.text)

        return PipelineResponse(
            query            = query,
            answer           = response.text,
            complexity       = complexity,
            cache_hit        = False,
            model_used       = response.model,
            classifier_s     = round(classifier_s, 2),
            compress_s       = round(compress_s, 2),
            generation_s     = round(generation_s, 2),
            total_s          = round(total_s, 2),
            input_tokens     = response.prompt_eval_count,
            output_tokens    = response.eval_count,
            compressed_query = compressed_query,
        )
