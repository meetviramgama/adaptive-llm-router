"""
Cloud Pipeline
--------------
Adaptive LLM Routing System using Groq API.

Architecture:

    User Query
        |
        v
    Semantic Cache
        |
        |-- HIT ------> Cached Answer
        |
        |-- MISS
              |
              v
        GPT-OSS-20B
        Classifier
              |
        +-----+-----+
        |           |
      SIMPLE      COMPLEX
        |           |
        v           v
    GPT-OSS-20B  GPT-OSS-20B
    Final Answer Compression
                        |
                        v
                  GPT-OSS-120B
                  Final Answer

Components:
    Classifier : GPT-OSS-20B
    Compressor : GPT-OSS-20B
    Simple ans : GPT-OSS-20B
    Complex ans: GPT-OSS-120B
    Cache      : FAISS + embedding API
"""

import os
import time
from dataclasses import dataclass

from groq_client import (
    generate,
    LOW_MODEL,
    HIGH_MODEL,
)


# ============================================================================
# Classifier prompt
# ============================================================================

CLASSIFIER_PROMPT = """You are a binary classifier.

Read the user query and classify it into exactly one category.

Reply COMPLEX if the query requires:
- explanation
- analysis
- comparison
- reasoning
- multiple steps
- detailed synthesis

Reply SIMPLE if the query requires:
- a short factual answer
- a simple definition
- a basic lookup-style response
- a short direct answer

Reply with ONE word only:

SIMPLE
or
COMPLEX
"""


# ============================================================================
# Compression prompt
# ============================================================================

COMPRESS_SYSTEM = """You are a query compression assistant.

Rewrite the complex user query into a precise structured brief.

Output ONLY these three things:

1. INTENT: What exactly is being asked (one sentence)
2. CONSTRAINTS: Key requirements or context (one sentence)
3. OUTPUT: What format or depth of answer is expected (one sentence)

Maximum 3 sentences.

Do not answer the query.
Do not add padding.
"""


# ============================================================================
# Response dataclass
# ============================================================================

@dataclass
class PipelineResponse:
    query: str
    answer: str
    complexity: str
    cache_hit: bool
    model_used: str

    classifier_s: float
    compress_s: float
    generation_s: float
    total_s: float

    input_tokens: int
    output_tokens: int

    compressed_query: str


# ============================================================================
# Semantic Cache
# ============================================================================

class SimpleCache:
    """
    Lightweight in-memory semantic cache.

    Uses:
        FAISS
        Embedding API

    The cache is intentionally in-memory for HuggingFace Spaces.
    """

    def __init__(self, threshold: float = 0.85):

        self.threshold = threshold

        # Each entry:
        # (query, answer)
        self.entries = []

        self._ready = False

        self._init()

    # ------------------------------------------------------------------------
    # Initialize FAISS
    # ------------------------------------------------------------------------

    def _init(self):

        try:

            import faiss
            import numpy as np

            self._faiss = faiss
            self._np = np

            # Nomic embedding dimension.
            self._index = faiss.IndexFlatIP(1536)

            self._ready = True

        except Exception as e:

            print(f"Cache initialization failed: {e}")

            self._ready = False

    # ------------------------------------------------------------------------
    # Generate embedding
    # ------------------------------------------------------------------------

    def _embed(self, text: str):

        if not self._ready:
            return None

        try:

            from groq import Groq

            api_key = os.environ.get("GROQ_API_KEY")

            if not api_key:
                return None

            client = Groq(api_key=api_key)

            response = client.embeddings.create(
                model="nomic-embed-text-v1.5",
                input=text,
            )

            vector = self._np.array(
                [response.data[0].embedding],
                dtype=self._np.float32,
            )

            self._faiss.normalize_L2(vector)

            return vector

        except Exception as e:

            print(f"Embedding error: {e}")

            return None

    # ------------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------------

    def lookup(self, query: str):

        if not self._ready:
            return None

        if self._index.ntotal == 0:
            return None

        vector = self._embed(query)

        if vector is None:
            return None

        scores, indices = self._index.search(
            vector,
            k=1,
        )

        similarity = float(scores[0][0])

        if similarity >= self.threshold:

            index = int(indices[0][0])

            return self.entries[index][1]

        return None

    # ------------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------------

    def store(self, query: str, answer: str):

        if not self._ready:
            return

        vector = self._embed(query)

        if vector is None:
            return

        self._index.add(vector)

        self.entries.append(
            (
                query,
                answer,
            )
        )

    # ------------------------------------------------------------------------
    # Clear
    # ------------------------------------------------------------------------

    def clear(self):

        if not self._ready:
            return

        self._index = self._faiss.IndexFlatIP(1536)

        self.entries = []

    # ------------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------------

    def stats(self):

        return {
            "total_entries": len(self.entries),
            "threshold": self.threshold,
        }


# ============================================================================
# Cloud Pipeline
# ============================================================================

class Pipeline:

    def __init__(
        self,
        use_cache: bool = True,
        use_compression: bool = True,
    ):

        self.use_cache = use_cache
        self.use_compression = use_compression

        self.cache = (
            SimpleCache()
            if use_cache
            else None
        )

    # ========================================================================
    # Classification
    # ========================================================================

    def _classify(self, query: str):

        response = generate(
            model=LOW_MODEL,
            prompt=query,
            system=CLASSIFIER_PROMPT,
            max_tokens=10,
            temperature=0.0,
        )

        raw = response.text.strip().upper()

        # Strictly detect COMPLEX.
        if raw == "COMPLEX" or "COMPLEX" in raw:
            label = "COMPLEX"
        else:
            label = "SIMPLE"

        return (
            label,
            response.wall_clock_s,
        )

    # ========================================================================
    # Compression
    # ========================================================================

    def _compress(self, query: str):

        response = generate(
            model=LOW_MODEL,
            prompt=query,
            system=COMPRESS_SYSTEM,
            max_tokens=200,
            temperature=0.3,
        )

        return (
            response.text,
            response.wall_clock_s,
        )

    # ========================================================================
    # Main answer function
    # ========================================================================

    def answer(self, query: str):

        start = time.time()

        # ====================================================================
        # Step 1: Semantic Cache
        # ====================================================================

        if self.use_cache and self.cache:

            cached = self.cache.lookup(query)

            if cached:

                return PipelineResponse(

                    query=query,

                    answer=cached,

                    complexity="CACHED",

                    cache_hit=True,

                    model_used="cache",

                    classifier_s=0.0,

                    compress_s=0.0,

                    generation_s=0.0,

                    total_s=round(
                        time.time() - start,
                        3,
                    ),

                    input_tokens=0,

                    output_tokens=0,

                    compressed_query="",
                )

        # ====================================================================
        # Step 2: Classify
        # ====================================================================

        complexity, classifier_s = self._classify(query)

        # ====================================================================
        # Step 3: Route
        # ====================================================================

        gen_start = time.time()

        compressed_query = ""

        compress_s = 0.0

        # --------------------------------------------------------------------
        # SIMPLE → GPT-OSS-20B
        # --------------------------------------------------------------------

        if complexity == "SIMPLE":

            response = generate(

                model=LOW_MODEL,

                prompt=query,

                system=(
                    "You are a helpful assistant. "
                    "Answer clearly and concisely."
                ),

                max_tokens=500,

                temperature=0.7,
            )

        # --------------------------------------------------------------------
        # COMPLEX → GPT-OSS-20B compression → GPT-OSS-120B
        # --------------------------------------------------------------------

        else:

            if self.use_compression:

                (
                    compressed_query,
                    compress_s,
                ) = self._compress(query)

                prompt_for_high = compressed_query

            else:

                prompt_for_high = query

            response = generate(

                model=HIGH_MODEL,

                prompt=prompt_for_high,

                system=(
                    "You are a helpful assistant. "
                    "Answer thoroughly and accurately."
                ),

                max_tokens=2000,

                temperature=0.7,
            )

        # ====================================================================
        # Timing
        # ====================================================================

        generation_s = time.time() - gen_start

        total_s = time.time() - start

        # ====================================================================
        # Step 4: Store in cache
        # ====================================================================

        if self.use_cache and self.cache:

            self.cache.store(
                query,
                response.text,
            )

        # ====================================================================
        # Return
        # ====================================================================

        return PipelineResponse(

            query=query,

            answer=response.text,

            complexity=complexity,

            cache_hit=False,

            model_used=response.model,

            classifier_s=round(
                classifier_s,
                2,
            ),

            compress_s=round(
                compress_s,
                2,
            ),

            generation_s=round(
                generation_s,
                2,
            ),

            total_s=round(
                total_s,
                2,
            ),

            input_tokens=response.prompt_eval_count,

            output_tokens=response.eval_count,

            compressed_query=compressed_query,
        )