"""
Adaptive LLM Routing Pipeline
-----------------------------

Flow:

                    USER QUERY
                        |
                        v
                 SEMANTIC CACHE
                   /        \
                HIT          MISS
                 |             |
                 v             v
             CACHE       COMPLEXITY ROUTER
                              |
                       +------+------+
                       |             |
                    SIMPLE        COMPLEX
                       |             |
                       v             v
                    20B          COMPRESS
                                     |
                                     v
                                   120B
                                     |
                                     v
                                   ANSWER
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
# CLASSIFIER
# ============================================================================

CLASSIFIER_SYSTEM = """
You are a strict query complexity classifier.

Classify the user's query into exactly one category:

SIMPLE
COMPLEX

SIMPLE means:
- Short factual questions
- Definitions
- Simple calculations
- Direct factual answers
- Basic knowledge questions

Examples:
What is Python?
What is the capital of France?
Who invented the telephone?
What is 2 + 2?
Define inflation.

COMPLEX means:
- Explanation
- Analysis
- Comparison
- Reasoning
- Strategy
- Recommendations
- Risks
- Multiple factors
- Cause and effect
- Detailed technical questions
- Step-by-step questions
- Evaluation

Examples:
Explain how transformers work.
Why does inflation affect interest rates?
Compare Docker and virtual machines.
Analyze the risks of prop trading.
Explain how gold prices are determined.
How can I optimize an LLM routing system?

IMPORTANT:
Return ONLY:

SIMPLE

or

COMPLEX
"""


# ============================================================================
# COMPRESSION
# ============================================================================

COMPRESSION_SYSTEM = """
You are a query compression assistant.

Convert the user's complex query into a precise structured brief.

Output ONLY:

INTENT: What exactly is being asked.
CONSTRAINTS: Important requirements or context.
OUTPUT: Expected answer format or depth.

Maximum 3 sentences.

Do NOT answer the question.
"""


# ============================================================================
# RESPONSE
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
    Local semantic cache.

    Uses:
        sentence-transformers
        FAISS

    This avoids relying on a Groq embeddings endpoint.
    """

    def __init__(self, threshold: float = 0.85):

        self.threshold = threshold

        self.entries = []

        self._ready = False

        self._model = None
        self._faiss = None
        self._np = None
        self._index = None

        self._init()


    # ------------------------------------------------------------------------
    # INITIALIZE
    # ------------------------------------------------------------------------

    def _init(self):

        try:

            import numpy as np
            import faiss

            from sentence_transformers import SentenceTransformer

            self._np = np
            self._faiss = faiss

            print("Loading semantic embedding model...")

            self._model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2"
            )

            dimension = 384

            self._index = faiss.IndexFlatIP(
                dimension
            )

            self._ready = True

            print("Semantic cache ready.")

        except Exception as e:

            print(
                f"Semantic cache unavailable: {e}"
            )

            self._ready = False


    # ------------------------------------------------------------------------
    # EMBEDDING
    # ------------------------------------------------------------------------

    def _embed(self, text: str):

        if not self._ready:

            return None

        try:

            vector = self._model.encode(
                [text],
                normalize_embeddings=True,
                convert_to_numpy=True,
            )

            return vector.astype(
                self._np.float32
            )

        except Exception as e:

            print(
                f"Embedding error: {e}"
            )

            return None


    # ------------------------------------------------------------------------
    # LOOKUP
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
            1,
        )

        similarity = float(
            scores[0][0]
        )

        if similarity >= self.threshold:

            idx = int(
                indices[0][0]
            )

            if 0 <= idx < len(self.entries):

                return self.entries[idx][1]

        return None


    # ------------------------------------------------------------------------
    # STORE
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
    # CLEAR
    # ------------------------------------------------------------------------

    def clear(self):

        if not self._ready:

            self.entries = []

            return

        self._index = self._faiss.IndexFlatIP(
            384
        )

        self.entries = []


    # ------------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------------

    def stats(self):

        return {
            "total_entries": len(
                self.entries
            ),
            "threshold": self.threshold,
            "ready": self._ready,
        }


# ============================================================================
# PIPELINE
# ============================================================================

class Pipeline:

    def __init__(
        self,
        use_cache=True,
        use_compression=True,
    ):

        self.use_cache = use_cache

        self.use_compression = use_compression

        self.cache = (
            SimpleCache()
            if use_cache
            else None
        )


    # ========================================================================
    # DETERMINISTIC COMPLEXITY
    # ========================================================================

    def _obvious_complexity(
        self,
        query: str,
    ):

        q = query.lower().strip()

        complex_patterns = [

            "explain ",
            "explain how",
            "explain why",

            "why ",

            "how does ",
            "how do ",
            "how can ",
            "how is ",
            "how are ",

            "compare ",
            "comparison",

            "analyze ",
            "analysis",

            "evaluate ",
            "evaluation",

            "advantages",
            "disadvantages",

            "pros and cons",

            "risks of",
            "risk of",

            "impact of",
            "effects of",
            "causes of",

            "strategy",
            "strategies",

            "step by step",
            "step-by-step",

            "difference between",

            "versus",
            " vs ",

            "trade-off",
            "tradeoffs",
            "trade offs",

            "recommend ",
            "recommendation",

            "best approach",
            "best strategy",
        ]

        for pattern in complex_patterns:

            if pattern in q:

                return "COMPLEX"

        return None


    # ========================================================================
    # CLASSIFIER
    # ========================================================================

    def _classify(
        self,
        query: str,
    ):

        # ------------------------------------------------------------
        # First use deterministic routing
        # ------------------------------------------------------------

        obvious = self._obvious_complexity(
            query
        )

        if obvious:

            return obvious, 0.0


        # ------------------------------------------------------------
        # GPT-OSS-20B classifier
        # ------------------------------------------------------------

        start = time.perf_counter()

        response = generate(
            model=LOW_MODEL,

            prompt=query,

            system=CLASSIFIER_SYSTEM,

            max_tokens=5,

            temperature=0.0,
        )

        classifier_time = (
            time.perf_counter()
            - start
        )


        raw = (
            response.text
            .strip()
            .upper()
        )


        if raw.startswith("COMPLEX"):

            label = "COMPLEX"

        elif raw.startswith("SIMPLE"):

            label = "SIMPLE"

        else:

            # Safe fallback
            label = "COMPLEX"


        return (
            label,
            classifier_time,
        )


    # ========================================================================
    # COMPRESSION
    # ========================================================================

    def _compress(
        self,
        query: str,
    ):

        response = generate(

            model=LOW_MODEL,

            prompt=query,

            system=COMPRESSION_SYSTEM,

            max_tokens=200,

            temperature=0.2,
        )

        return (
            response.text,
            response.wall_clock_s,
        )


    # ========================================================================
    # ANSWER
    # ========================================================================

    def answer(
        self,
        query: str,
    ):

        total_start = time.perf_counter()


        # ====================================================================
        # CACHE
        # ====================================================================

        if self.use_cache and self.cache:

            cached = self.cache.lookup(
                query
            )

            if cached:

                total_time = (
                    time.perf_counter()
                    - total_start
                )

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
                        total_time,
                        3,
                    ),

                    input_tokens=0,

                    output_tokens=0,

                    compressed_query="",
                )


        # ====================================================================
        # CLASSIFICATION
        # ====================================================================

        complexity, classifier_s = (
            self._classify(query)
        )


        # ====================================================================
        # ROUTING
        # ====================================================================

        compressed_query = ""

        compress_s = 0.0

        generation_start = (
            time.perf_counter()
        )


        # --------------------------------------------------------------------
        # SIMPLE → GPT-OSS-20B
        # --------------------------------------------------------------------

        if complexity == "SIMPLE":

            response = generate(

                model=LOW_MODEL,

                prompt=query,

                system="""
You are a helpful assistant.

Answer the user's question directly,
accurately, and clearly.

For simple questions, keep the answer
concise and avoid unnecessary explanation.
""",

                max_tokens=500,

                temperature=0.7,
            )


        # --------------------------------------------------------------------
        # COMPLEX → 20B COMPRESSION → 120B
        # --------------------------------------------------------------------

        else:

            if self.use_compression:

                compressed_query, compress_s = (
                    self._compress(query)
                )

                prompt_for_high = (
                    compressed_query
                )

                high_system = """
You are a highly capable reasoning assistant.

The user query has been compressed into a
structured brief by a smaller model.

Use the compressed brief to understand the
original intent and provide a thorough,
accurate and useful answer.

Do not mention the compression process.
"""

            else:

                prompt_for_high = query

                high_system = """
You are a highly capable assistant.

Answer the user's question thoroughly,
accurately and clearly.
"""


            response = generate(

                model=HIGH_MODEL,

                prompt=prompt_for_high,

                system=high_system,

                max_tokens=2000,

                temperature=0.7,
            )


        # ====================================================================
        # TIMING
        # ====================================================================

        generation_s = (
            time.perf_counter()
            - generation_start
        )


        total_s = (
            time.perf_counter()
            - total_start
        )


        # ====================================================================
        # CACHE STORE
        # ====================================================================

        if (
            self.use_cache
            and self.cache
            and response.text
        ):

            self.cache.store(

                query,

                response.text,
            )


        # ====================================================================
        # RESPONSE
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