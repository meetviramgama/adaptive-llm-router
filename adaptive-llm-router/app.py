"""
HuggingFace Spaces Entry Point
-------------------------------
Adaptive LLM Routing System — Cloud Demo

Architecture:

    Simple Query
        → GPT-OSS-20B

    Complex Query
        → GPT-OSS-20B compression
        → GPT-OSS-120B answer

    Repeated Query
        → Semantic Cache

Setup:

    1. Upload files to HuggingFace Space
    2. Add GROQ_API_KEY in:
       Space Settings → Secrets
    3. Space automatically deploys app.py
"""

import os

import streamlit as st

from pipeline_cloud import Pipeline
from groq_client import LOW_MODEL, HIGH_MODEL


# ============================================================================
# Page configuration
# ============================================================================

st.set_page_config(
    page_title="Adaptive LLM Router",
    page_icon="⚡",
    layout="wide",
)


# ============================================================================
# Benchmark Configuration
# ============================================================================

# IMPORTANT:
# Set this to the measured average latency of your
# ALWAYS-HIGH-MODEL baseline.
#
# Do not reuse an old Qwen/Ollama benchmark here.
#
# Replace this value after benchmarking GPT-OSS-120B.
BASELINE_AVG = 69.03

BASELINE_MODEL = HIGH_MODEL


# ============================================================================
# Compute weight
# ============================================================================

def get_weight(model: str) -> float:

    """
    Approximate compute weight based on model size.

    GPT-OSS-20B  → 20
    GPT-OSS-120B → 120

    NOTE:
    This is a simple proxy, NOT actual Groq billing cost.
    """

    model_lower = model.lower()

    if "20b" in model_lower:
        return 20.0

    if "120b" in model_lower:
        return 120.0

    return 0.0


# ============================================================================
# Session State
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
# Load Pipeline
# ============================================================================

@st.cache_resource
def load_pipeline():

    return Pipeline(
        use_cache=True,
        use_compression=True,
    )


# ============================================================================
# API Key Check
# ============================================================================

if not os.environ.get("GROQ_API_KEY"):

    st.error(
        "⚠️ GROQ_API_KEY is not set. "
        "Add it in HuggingFace Space → "
        "Settings → Secrets."
    )

    st.stop()


# ============================================================================
# Header
# ============================================================================

st.title("⚡ Adaptive LLM Routing System")

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
# Sidebar
# ============================================================================

with st.sidebar:

    st.header("📊 Session Statistics")

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
    # Latency
    # ========================================================================

    st.subheader("⏱ Latency")

    if n > 0:

        avg = (
            st.session_state.total_full_s
            / n
        )

        pct = (
            1 - avg / BASELINE_AVG
        ) * 100

        st.metric(
            "Avg Latency",
            f"{avg:.2f}s",
            delta=f"{pct:.1f}% vs baseline",
            delta_color=(
                "normal"
                if pct > 0
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
    # Compute
    # ========================================================================

    st.subheader("💰 Compute Proxy")

    if (
        st.session_state.total_compute_base
        > 0
    ):

        reduction = (
            1
            - (
                st.session_state.total_compute_full
                / st.session_state.total_compute_base
            )
        ) * 100

        st.metric(
            "Compute Reduction",
            f"{reduction:.1f}%",
            delta="vs always GPT-OSS-120B",
        )

    st.caption(
        "Model-size weighted proxy; "
        "not actual API billing."
    )

    st.divider()

    # ========================================================================
    # Clear history
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
    # Clear cache
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
# Input
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
# Example Queries
# ============================================================================

st.markdown("**Try these:**")

ex_cols = st.columns(4)

examples = [

    "What is the capital of France?",

    "Who invented the telephone?",

    "Explain how gold prices are determined in global markets.",

    "What are the risks of prop trading without risk management?",
]

for col, example in zip(
    ex_cols,
    examples,
):

    with col:

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
# Process Query
# ============================================================================

if submit and query.strip():

    pipeline = load_pipeline()

    with st.spinner(
        "Analyzing and routing query..."
    ):

        result = pipeline.answer(
            query.strip()
        )

    st.divider()

    # ========================================================================
    # Routing Status
    # ========================================================================

    if result.cache_hit:

        st.success(
            "⚡ CACHE HIT — "
            "Reused cached answer without LLM generation."
        )

    elif result.complexity == "SIMPLE":

        st.info(
            "🟢 **SIMPLE** → "
            f"GPT-OSS-20B"
        )

    else:

        st.warning(
            "🟠 **COMPLEX** → "
            "GPT-OSS-20B compression → "
            "GPT-OSS-120B answer"
        )

    # ========================================================================
    # Metrics
    # ========================================================================

    m1, m2, m3, m4, m5, m6 = st.columns(6)

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

    model_display = result.model_used

    if model_display == "cache":

        model_display = "CACHE"

    elif len(model_display) > 18:

        model_display = (
            "GPT-OSS-20B"
            if "20b" in model_display.lower()
            else "GPT-OSS-120B"
        )

    m5.metric(
        "Model",
        model_display,
    )

    saving = max(
        0,
        BASELINE_AVG - result.total_s,
    )

    m6.metric(
        "Time Saved",
        f"{saving:.1f}s",
        delta=(
            f"{saving / BASELINE_AVG * 100:.0f}%"
        ),
    )

    # ========================================================================
    # Compressed Query
    # ========================================================================

    if result.compressed_query:

        st.subheader(
            "🔍 Compressed Brief"
        )

        st.info(
            result.compressed_query
        )

        st.caption(
            "GPT-OSS-20B compressed the original query. "
            "GPT-OSS-120B generated the final answer "
            "from the compressed brief."
        )

    # ========================================================================
    # Answer
    # ========================================================================

    st.subheader("💬 Answer")

    st.markdown(
        result.answer
    )

    # ========================================================================
    # Compute
    # ========================================================================

    tokens = (
        (result.input_tokens or 0)
        +
        (result.output_tokens or 0)
    )

    if result.cache_hit:

        compute_full = 0.0

    else:

        compute_full = (
            tokens
            * get_weight(
                result.model_used
            )
        )

    compute_base = (
        tokens
        * get_weight(
            BASELINE_MODEL
        )
    )

    # ========================================================================
    # Session totals
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
    # History
    # ========================================================================

    st.session_state.history.append(

        {

            "query": query.strip(),

            "complexity": result.complexity,

            "cache_hit": result.cache_hit,

            "model_used": result.model_used,

            "total_s": result.total_s,

            "classifier_s": result.classifier_s,

            "compress_s": result.compress_s,

            "generation_s": result.generation_s,

            "compressed_query": result.compressed_query,

            "answer": result.answer[:300],

            "compute_units": compute_full,

        }
    )


# ============================================================================
# Query History
# ============================================================================

if st.session_state.history:

    st.divider()

    st.subheader(
        "📋 Query History"
    )

    for i, h in enumerate(
        reversed(
            st.session_state.history
        ),
        1,
    ):

        icon = (

            "⚡"
            if h["cache_hit"]

            else (
                "🟢"
                if h["complexity"] == "SIMPLE"
                else "🟠"
            )
        )

        with st.expander(

            (
                f"{icon} "
                f"[{i}] "
                f"{h['query'][:60]} "
                f"— {h['total_s']:.1f}s"
            ),

            expanded=(i == 1),
        ):

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Complexity",
                h["complexity"],
            )

            c2.metric(
                "Cache Hit",
                "Yes"
                if h["cache_hit"]
                else "No",
            )

            c3.metric(
                "Compress",
                f"{h['compress_s']:.2f}s",
            )

            c4.metric(
                "Compute Proxy",
                f"{h['compute_units']:.0f}",
            )

            if h["compressed_query"]:

                st.markdown(
                    "**Compressed brief:**"
                )

                st.info(
                    h["compressed_query"]
                )

            st.markdown(
                "**Answer:**"
            )

            st.markdown(
                h["answer"]
                + (
                    "..."
                    if len(h["answer"]) == 300
                    else ""
                )
            )