"""
Adaptive LLM Routing Pipeline
------------------------------

Routing:

    Query
      |
      v
    Semantic Cache
      |
      |-- HIT --> Cached Answer
      |
      |-- MISS
            |
            v
      Complexity Router
            |
       +----+----+
       |         |
     SIMPLE   COMPLEX
       |         |
       v         v
     20B       20B
       |      Compress
       |         |
       |         v
       |       120B
       |         |
       +----+----+
            |
            v
          Answer
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

CLASSIFIER_PROMPT = """You are a strict binary query complexity classifier.

Classify the user query as exactly one:

SIMPLE
COMPLEX

SIMPLE:
Use SIMPLE only for short factual questions that need a direct answer.

Examples:
- What is the capital of France?
- Who invented the telephone?
- What is Python?
- What is 2 + 2?
- Define inflation.

COMPLEX:
Use COMPLEX when the query requires explanation, analysis,
comparison, reasoning, multiple factors, causes/effects,
recommendations, strategy, risks, evaluation, or detailed synthesis.

Examples:
- Explain how gold prices are determined in global markets.
- Why does inflation affect interest rates?
- Compare React and Angular.
- Analyze the risks of prop trading.
- Explain how transformers work.
- How does monetary policy affect gold prices?
- Compare Docker and virtual machines.

Return exactly one word:

SIMPLE

or

COMPLEX
"""


# ============================================================================
# COMPRESSION PROMPT
# ============================================================================

COMPRESS_SYSTEM = """You are a query compression assistant.

Rewrite the user's complex query into a precise structured brief.

Output ONLY:

INTENT: What exactly is being asked.
CONSTRAINTS: Important requirements or context.
OUTPUT: Expected answer format or depth.

Maximum 3 sentences.

Do not answer the query.
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

    def __init__(self, threshold: float = 0.85):

        self.threshold = threshold
        self.entries = []
        self._ready = False

        self._init()

    def _init(self):

        try:

            import faiss
            import numpy as np

            self._faiss = faiss
            self._np = np

            self._index = faiss.IndexFlatIP(1536)

            self._ready = True

        except Exception as e:

            print(f"FAISS cache unavailable: {e}")

            self._ready = False

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
                [response.data[0].embedding],
                dtype=self._np.float32,
            )

            self._faiss.normalize_L2(vector)

            return vector

        except Exception as e:

            print(f"Embedding error: {e}")

            return None

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

            idx = int(indices[0][0])

            return self.entries[idx][1]

        return None

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
            (query, answer)
        )

    def clear(self):

        if not self._ready:
            return

        self._index = (
            self._faiss.IndexFlatIP(1536)
        )

        self.entries = []

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
    # DETERMINISTIC COMPLEXITY CHECK
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

        # --------------------------------------------------------------------
        # First: deterministic routing for obvious complex queries
        # --------------------------------------------------------------------

        obvious = self._obvious_complexity(
            query
        )

        if obvious:

            return obvious, 0.0

        # --------------------------------------------------------------------
        # Otherwise use GPT-OSS-20B classifier
        # --------------------------------------------------------------------

        response = generate(

            model=LOW_MODEL,

            prompt=query,

            system=CLASSIFIER_PROMPT,

            max_tokens=5,

            temperature=0.0,
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
            response.wall_clock_s,
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

            system=COMPRESS_SYSTEM,

            max_tokens=200,

            temperature=0.3,
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

        start = time.time()

        # ====================================================================
        # CACHE
        # ====================================================================

        if self.use_cache and self.cache:

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
        # CLASSIFICATION
        # ====================================================================

        complexity, classifier_s = (
            self._classify(query)
        )

        # ====================================================================
        # ROUTING
        # ====================================================================

        generation_start = time.time()

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
        # COMPLEX → 20B → 120B
        # --------------------------------------------------------------------

        else:

            if self.use_compression:

                (
                    compressed_query,
                    compress_s,
                ) = self._compress(query)

                prompt_for_high = (
                    compressed_query
                )

            else:

                prompt_for_high = query

            response = generate(

                model=HIGH_MODEL,

                prompt=prompt_for_high,

                system=(
                    "You are a highly capable assistant. "
                    "Answer thoroughly and accurately."
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
        # CACHE STORE
        # ====================================================================

        if self.use_cache and self.cache:

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

            compressed_query=compressed_query,
        )