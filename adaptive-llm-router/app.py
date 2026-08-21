"""
HuggingFace Spaces Entry Point
-------------------------------
Adaptive LLM Routing System — Cloud Demo

Architecture:

    SIMPLE
        User Query
            ↓
        GPT-OSS-20B
            ↓
        Answer

    COMPLEX
        User Query
            ↓
        GPT-OSS-20B Classifier
            ↓
        GPT-OSS-20B Compression
            ↓
        GPT-OSS-120B
            ↓
        Answer

    REPEATED
        User Query
            ↓
        Semantic Cache
            ↓
        Cached Answer
"""


import os

import streamlit as st

from pipeline_cloud import Pipeline
from groq_client import (
    LOW_MODEL,
    HIGH_MODEL,
)


# ============================================================================
# PAGE CONFIG
# ============================================================================

st.set_page_config(
    page_title="Adaptive LLM Router",
    page_icon="⚡",
    layout="wide",
)


# ============================================================================
# MODEL / BASELINE CONFIGURATION
# ============================================================================

LOW_MODEL_NAME = LOW_MODEL
HIGH_MODEL_NAME = HIGH_MODEL

# IMPORTANT:
# Replace this with your ACTUAL measured baseline
# for always using GPT-OSS-120B.
#
# If 69.03s is from your old Qwen/Ollama experiment,
# do NOT present it as a GPT-OSS-120B benchmark.
BASELINE_AVG = 69.03

BASELINE_MODEL = HIGH_MODEL


# ============================================================================
# COMPUTE WEIGHT
# ============================================================================

def get_weight(model: str) -> float:

    """
    Model-size weighted compute proxy.

    GPT-OSS-20B  → 20
    GPT-OSS-120B → 120

    This is only a proxy.
    It is NOT actual Groq billing cost.
    """

    model_lower = model.lower()

    if "120b" in model_lower:
        return 120.0

    if "20b" in model_lower:
        return 20.0

    return 0.0


# ============================================================================
# SESSION STATE
# ============================================================================

if "history" not in st.session_state:

    st.session_state.history = []


if "total_full_s" not in st.session_state:

    st.session_state.total_full_s = 0.0


if "total_compute_full" not in st.session_state:

    st.session_state.total_compute_full = 0.0


if "total_compute_base" not in st.session_state:

    st.session_state.total_compute_base = 0.0


# ============================================================================
# LOAD PIPELINE
# ============================================================================

@st.cache_resource
def load_pipeline():

    return Pipeline(
        use_cache=True,
        use_compression=True,
    )


# ============================================================================
# API KEY CHECK
# ============================================================================

if not os.environ.get(
    "GROQ_API_KEY"
):

    st.error(
        "⚠️ GROQ_API_KEY is not set. "
        "Go to HuggingFace Space → "
        "Settings → Secrets and add GROQ_API_KEY."
    )

    st.stop()


# ============================================================================
# HEADER
# ============================================================================

st.title(
    "⚡ Adaptive LLM Routing System"
)

st.markdown(
    """
Routes each query to the appropriate model automatically.

**Simple queries** → GPT-OSS-20B  
**Complex queries** → GPT-OSS-20B compression → GPT-OSS-120B  
**Repeated queries** → Semantic Cache
"""
)

st.divider()


# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:

    st.header(
        "📊 Session Statistics"
    )

    # ------------------------------------------------------------------------
    # Current session statistics
    # ------------------------------------------------------------------------

    n = len(
        st.session_state.history
    )

    cache_hits = sum(
        1
        for h in st.session_state.history
        if h["cache_hit"]
    )

    simple_count = sum(
        1
        for h in st.session_state.history
        if h["complexity"] == "SIMPLE"
    )

    complex_count = sum(
        1
        for h in st.session_state.history
        if h["complexity"] == "COMPLEX"
    )

    compressed = sum(
        1
        for h in st.session_state.history
        if h["compressed_query"]
    )

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    st.metric(
        "Total Queries",
        n,
    )

    st.metric(
        "Cache Hits",
        cache_hits,
        delta=(
            f"{cache_hits / n * 100:.0f}%"
            if n
            else None
        ),
    )

    st.metric(
        "Simple → GPT-OSS-20B",
        simple_count,
    )

    st.metric(
        "Complex → GPT-OSS-120B",
        complex_count,
    )

    st.metric(
        "Compressed",
        compressed,
    )

    st.divider()

    # ========================================================================
    # LATENCY
    # ========================================================================

    st.subheader(
        "⏱ Latency"
    )

    if n > 0:

        average_latency = (
            st.session_state.total_full_s
            / n
        )

        percentage = (
            1
            - (
                average_latency
                / BASELINE_AVG
            )
        ) * 100

        st.metric(
            "Avg Latency",
            f"{average_latency:.2f}s",
            delta=(
                f"{percentage:.1f}% vs baseline"
            ),
            delta_color=(
                "normal"
                if percentage > 0
                else "inverse"
            ),
        )

    st.metric(
        "Baseline",
        f"{BASELINE_AVG:.2f}s",
    )

    st.caption(
        f"Baseline: always {BASELINE_MODEL}"
    )

    st.divider()

    # ========================================================================
    # COMPUTE
    # ========================================================================

    st.subheader(
        "💰 Compute Proxy"
    )

    if (
        st.session_state.total_compute_base
        > 0
    ):

        compute_reduction = (
            1
            - (
                st.session_state.total_compute_full
                / st.session_state.total_compute_base
            )
        ) * 100

        st.metric(
            "Compute Reduction",
            f"{compute_reduction:.1f}%",
            delta=(
                "vs always GPT-OSS-120B"
            ),
        )

    st.caption(
        "Model-size weighted proxy; "
        "not actual API billing."
    )

    st.divider()

    # ========================================================================
    # CLEAR HISTORY
    # ========================================================================

    if st.button(
        "🗑 Clear History",
        use_container_width=True,
    ):

        st.session_state.history = []

        st.session_state.total_full_s = 0.0

        st.session_state.total_compute_full = 0.0

        st.session_state.total_compute_base = 0.0

        st.rerun()

    # ========================================================================
    # CLEAR CACHE
    # ========================================================================

    if st.button(
        "🧹 Clear Cache",
        use_container_width=True,
    ):

        pipeline = load_pipeline()

        if pipeline.cache:

            pipeline.cache.clear()

        st.success(
            "Cache cleared."
        )


# ============================================================================
# QUERY INPUT
# ============================================================================

col1, col2 = st.columns(
    [3, 1]
)


with col1:

    query = st.text_input(
        "Enter your query:",
        placeholder=(
            "e.g. What is the capital of France? "
            "/ Explain how gold prices work"
        ),
    )


with col2:

    st.markdown(
        "<br>",
        unsafe_allow_html=True,
    )

    submit = st.button(
        "🚀 Submit",
        use_container_width=True,
    )


# ============================================================================
# EXAMPLE QUERIES
# ============================================================================

st.markdown(
    "**Try these:**"
)

example_columns = st.columns(4)

examples = [

    "What is the capital of France?",

    "Who invented the telephone?",

    "Explain how gold prices are determined in global markets.",

    "What are the risks of prop trading without risk management?",
]


for column, example in zip(
    example_columns,
    examples,
):

    with column:

        label = (
            example[:35] + "..."
            if len(example) > 35
            else example
        )

        if st.button(
            label,
            use_container_width=True,
        ):

            query = example

            submit = True


# ============================================================================
# PROCESS QUERY
# ============================================================================

if submit and query.strip():

    clean_query = query.strip()

    pipeline = load_pipeline()

    with st.spinner(
        "Analyzing query and selecting model..."
    ):

        result = pipeline.answer(
            clean_query
        )

    # ========================================================================
    # ROUTING STATUS
    # ========================================================================

    st.divider()

    if result.cache_hit:

        st.success(
            "⚡ **CACHE HIT** — "
            "Reused cached answer without "
            "LLM generation."
        )

    elif result.complexity == "SIMPLE":

        st.info(
            "🟢 **SIMPLE** → "
            "**GPT-OSS-20B**"
        )

    else:

        st.warning(
            "🟠 **COMPLEX** → "
            "**GPT-OSS-20B** compression "
            "→ **GPT-OSS-120B** answer"
        )

    # ========================================================================
    # METRICS
    # ========================================================================

    m1, m2, m3, m4, m5, m6 = (
        st.columns(6)
    )

    m1.metric(
        "Total",
        f"{result.total_s:.2f}s",
    )

    m2.metric(
        "Classify",
        f"{result.classifier_s:.2f}s",
    )

    m3.metric(
        "Compress",
        f"{result.compress_s:.2f}s",
    )

    m4.metric(
        "Generate",
        f"{result.generation_s:.2f}s",
    )

    # ------------------------------------------------------------------------
    # Model display
    # ------------------------------------------------------------------------

    if result.cache_hit:

        model_display = "CACHE"

    elif "120b" in result.model_used.lower():

        model_display = "GPT-OSS-120B"

    elif "20b" in result.model_used.lower():

        model_display = "GPT-OSS-20B"

    else:

        model_display = result.model_used

    m5.metric(
        "Model",
        model_display,
    )

    # ------------------------------------------------------------------------
    # Time saved
    # ------------------------------------------------------------------------

    saving = max(
        0,
        BASELINE_AVG - result.total_s,
    )

    saving_percentage = (
        saving
        / BASELINE_AVG
        * 100
    )

    m6.metric(
        "Time Saved",
        f"{saving:.1f}s",
        delta=(
            f"{saving_percentage:.0f}%"
        ),
    )

    # ========================================================================
    # COMPRESSED QUERY
    # ========================================================================

    if result.compressed_query:

        st.subheader(
            "🔍 Compressed Brief"
        )

        st.info(
            result.compressed_query
        )

        st.caption(
            "GPT-OSS-20B compressed the original "
            "query. GPT-OSS-120B generated the "
            "final answer from the compressed brief."
        )

    # ========================================================================
    # ANSWER
    # ========================================================================

    st.subheader(
        "💬 Answer"
    )

    st.markdown(
        result.answer
    )

    # ========================================================================
    # COMPUTE PROXY
    # ========================================================================

    total_tokens = (
        (result.input_tokens or 0)
        +
        (result.output_tokens or 0)
    )

    if result.cache_hit:

        compute_full = 0.0

    else:

        compute_full = (
            total_tokens
            * get_weight(
                result.model_used
            )
        )

    # Always-high baseline
    compute_base = (
        total_tokens
        * get_weight(
            BASELINE_MODEL
        )
    )

    # ========================================================================
    # UPDATE SESSION TOTALS
    # ========================================================================

    st.session_state.total_full_s += (
        result.total_s
    )

    st.session_state.total_compute_full += (
        compute_full
    )

    st.session_state.total_compute_base += (
        compute_base
    )

    # ========================================================================
    # ADD TO HISTORY
    # ========================================================================

    st.session_state.history.append(
        {
            "query": clean_query,

            "complexity": result.complexity,

            "cache_hit": result.cache_hit,

            "model_used": result.model_used,

            "total_s": result.total_s,

            "classifier_s": result.classifier_s,

            "compress_s": result.compress_s,

            "generation_s": result.generation_s,

            "compressed_query": (
                result.compressed_query
            ),

            "answer": (
                result.answer[:300]
            ),

            "compute_units": (
                compute_full
            ),
        }
    )

    # ========================================================================
    # IMPORTANT:
    # Force Streamlit to rerun so that the sidebar
    # immediately displays the new statistics.
    # ========================================================================

    st.rerun()


# ============================================================================
# QUERY HISTORY
# ============================================================================

if st.session_state.history:

    st.divider()

    st.subheader(
        "📋 Query History"
    )

    for i, history_item in enumerate(
        reversed(
            st.session_state.history
        ),
        1,
    ):

        # --------------------------------------------------------------------
        # Icon
        # --------------------------------------------------------------------

        if history_item["cache_hit"]:

            icon = "⚡"

        elif (
            history_item["complexity"]
            == "SIMPLE"
        ):

            icon = "🟢"

        else:

            icon = "🟠"

        # --------------------------------------------------------------------
        # Expander
        # --------------------------------------------------------------------

        with st.expander(

            (
                f"{icon} "
                f"[{i}] "
                f"{history_item['query'][:60]} "
                f"— "
                f"{history_item['total_s']:.1f}s"
            ),

            expanded=(
                i == 1
            ),
        ):

            c1, c2, c3, c4 = (
                st.columns(4)
            )

            c1.metric(
                "Complexity",
                history_item[
                    "complexity"
                ],
            )

            c2.metric(
                "Cache Hit",
                (
                    "Yes"
                    if history_item[
                        "cache_hit"
                    ]
                    else "No"
                ),
            )

            c3.metric(
                "Compress",
                (
                    f"{history_item['compress_s']:.2f}s"
                ),
            )

            c4.metric(
                "Compute Proxy",
                (
                    f"{history_item['compute_units']:.0f}"
                ),
            )

            # ---------------------------------------------------------------
            # Compressed brief
            # ---------------------------------------------------------------

            if history_item[
                "compressed_query"
            ]:

                st.markdown(
                    "**Compressed brief:**"
                )

                st.info(
                    history_item[
                        "compressed_query"
                    ]
                )

            # ---------------------------------------------------------------
            # Answer
            # ---------------------------------------------------------------

            st.markdown(
                "**Answer:**"
            )

            answer_preview = (
                history_item["answer"]
            )

            if len(
                answer_preview
            ) == 300:

                answer_preview += "..."

            st.markdown(
                answer_preview
            )