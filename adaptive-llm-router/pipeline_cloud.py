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
        |-- HIT -----------------> Cached Answer
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


Models:
    Low tier  : GPT-OSS-20B
    High tier : GPT-OSS-120B
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
# CLASSIFIER PROMPT
# ============================================================================

CLASSIFIER_PROMPT = """You are a strict query complexity classifier.

Your job is to classify the USER QUERY into exactly one category:

SIMPLE
or
COMPLEX

============================================================
SIMPLE
============================================================

Return SIMPLE ONLY when the query can be answered with a short,
direct factual response.

Examples:

What is the capital of France?
→ SIMPLE

Who invented the telephone?
→ SIMPLE

What is Python?
→ SIMPLE

What is 2 + 2?
→ SIMPLE

Define inflation.
→ SIMPLE

When was Tesla founded?
→ SIMPLE

============================================================
COMPLEX
============================================================

Return COMPLEX when the query requires any meaningful:

- explanation
- analysis
- comparison
- reasoning
- multiple factors
- multiple steps
- causes and effects
- risks or trade-offs
- recommendations
- strategy
- detailed synthesis
- step-by-step instructions
- evaluation
- interpretation

IMPORTANT:

Queries containing words or phrases such as:

"explain"
"why"
"how"
"compare"
"analyze"
"evaluate"
"risks"
"advantages"
"disadvantages"
"strategy"
"impact"
"causes"
"effects"

should normally be classified as COMPLEX.

Examples:

Explain how gold prices are determined in global markets.
→ COMPLEX

Why does inflation affect interest rates?
→ COMPLEX

Compare React and Angular for a large application.
→ COMPLEX

What are the risks of prop trading without risk management?
→ COMPLEX

Analyze the advantages and disadvantages of remote work.
→ COMPLEX

Explain how transformers work.
→ COMPLEX

How does monetary policy affect gold prices?
→ COMPLEX

Compare Docker and virtual machines.
→ COMPLEX

============================================================
IMPORTANT
============================================================

Do NOT answer the query.

Do NOT explain your decision.

Return EXACTLY ONE WORD:

SIMPLE

or

COMPLEX
"""


# ============================================================================
# COMPRESSION PROMPT
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
# RESPONSE DATACLASS
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
# SEMANTIC CACHE
# ============================================================================

class SimpleCache:
    """
    Lightweight in-memory semantic cache.

    Uses:
        FAISS
        Embedding API

    The cache is kept in memory for HuggingFace Spaces.
    """

    def __init__(self, threshold: float = 0.85):

        self.threshold = threshold

        # Each item:
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

            # Nomic embedding dimension
            self._index = faiss.IndexFlatIP(1536)

            self._ready = True

        except Exception as e:

            print(
                f"Cache initialization failed: {e}"
            )

            self._ready = False

    # ------------------------------------------------------------------------
    # Generate embedding
    # ------------------------------------------------------------------------

    def _embed(self, text: str):

        if not self._ready:
            return None

        try:

            from groq import Groq

            api_key = os.environ.get(
                "GROQ_API_KEY"
            )

            if not api_key:
                return None

            client = Groq(
                api_key=api_key
            )

            response = client.embeddings.create(
                model="nomic-embed-text-v1.5",
                input=text,
            )

            vector = self._np.array(
                [
                    response.data[0].embedding
                ],
                dtype=self._np.float32,
            )

            self._faiss.normalize_L2(
                vector
            )

            return vector

        except Exception as e:

            print(
                f"Embedding error: {e}"
            )

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

        similarity = float(
            scores[0][0]
        )

        if similarity >= self.threshold:

            index = int(
                indices[0][0]
            )

            return self.entries[index][1]

        return None

    # ------------------------------------------------------------------------
    # Store
    # ------------------------------------------------------------------------

    def store(
        self,
        query: str,
        answer: str,
    ):

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

        self._index = (
            self._faiss.IndexFlatIP(1536)
        )

        self.entries = []

    # ------------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------------

    def stats(self):

        return {
            "total_entries": len(
                self.entries
            ),
            "threshold": self.threshold,
        }


# ============================================================================
# PIPELINE
# ============================================================================

class Pipeline:

    def __init__(
        self,
        use_cache: bool = True,
        use_compression: bool = True,
    ):

        self.use_cache = use_cache

        self.use_compression = (
            use_compression
        )

        self.cache = (
            SimpleCache()
            if use_cache
            else None
        )

    # ========================================================================
    # CLASSIFIER
    # ========================================================================

    def _classify(
        self,
        query: str,
    ):

        response = generate(
            model=LOW_MODEL,
            prompt=query,
            system=CLASSIFIER_PROMPT,
            max_tokens=10,
            temperature=0.0,
        )

        raw = (
            response.text
            .strip()
            .upper()
        )

        # --------------------------------------------------------------------
        # Strict classification
        # --------------------------------------------------------------------

        if raw.startswith("COMPLEX"):

            label = "COMPLEX"

        elif raw.startswith("SIMPLE"):

            label = "SIMPLE"

        else:

            # Safe fallback:
            # if classifier gives unexpected output,
            # route to powerful model.
            label = "COMPLEX"

        return (
            label,
            response.wall_clock_s,
        )

    # ========================================================================
    # COMPRESSOR
    # ========================================================================

    def _compress(
        self,
        query: str,
    ):

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
    # MAIN ANSWER FUNCTION
    # ========================================================================

    def answer(
        self,
        query: str,
    ):

        start = time.time()

        # ====================================================================
        # STEP 1 — SEMANTIC CACHE
        # ====================================================================

        if (
            self.use_cache
            and self.cache
        ):

            cached = self.cache.lookup(
                query
            )

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
        # STEP 2 — CLASSIFY
        # ====================================================================

        complexity, classifier_s = (
            self._classify(query)
        )

        # ====================================================================
        # STEP 3 — ROUTE
        # ====================================================================

        generation_start = time.time()

        compressed_query = ""

        compress_s = 0.0

        # ====================================================================
        # SIMPLE → GPT-OSS-20B
        # ====================================================================

        if complexity == "SIMPLE":

            response = generate(

                model=LOW_MODEL,

                prompt=query,

                system=(
                    "You are a helpful assistant. "
                    "Answer clearly, accurately, "
                    "and concisely."
                ),

                max_tokens=500,

                temperature=0.7,
            )

        # ====================================================================
        # COMPLEX → 20B COMPRESSION → 120B
        # ====================================================================

        else:

            if self.use_compression:

                (
                    compressed_query,
                    compress_s,
                ) = self._compress(
                    query
                )

                prompt_for_high = (
                    compressed_query
                )

            else:

                prompt_for_high = query

            response = generate(

                model=HIGH_MODEL,

                prompt=prompt_for_high,

                system=(
                    "You are a highly capable "
                    "assistant. Answer thoroughly, "
                    "accurately, and logically. "
                    "Provide useful detail while "
                    "staying focused on the user's "
                    "request."
                ),

                max_tokens=2000,

                temperature=0.7,
            )

        # ====================================================================
        # TIMING
        # ====================================================================

        generation_s = (
            time.time()
            - generation_start
        )

        total_s = (
            time.time()
            - start
        )

        # ====================================================================
        # STEP 4 — CACHE RESPONSE
        # ====================================================================

        if (
            self.use_cache
            and self.cache
        ):

            self.cache.store(
                query,
                response.text,
            )

        # ====================================================================
        # RETURN RESPONSE
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

            input_tokens=(
                response.prompt_eval_count
            ),

            output_tokens=(
                response.eval_count
            ),

            compressed_query=(
                compressed_query
            ),
        )