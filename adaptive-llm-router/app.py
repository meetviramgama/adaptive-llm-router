"""
HuggingFace Spaces Entry Point
-------------------------------
Adaptive LLM Routing System — Cloud Demo

This file is the root-level app.py for HuggingFace Spaces deployment.
It uses Groq API instead of local Ollama.

Setup:
    1. Upload all files to HuggingFace Space
    2. Add GROQ_API_KEY in Space Settings → Secrets
    3. Space will auto-deploy and run this file
"""

import os
import streamlit as st
from pipeline_cloud import Pipeline


# ============================================================================
# Page config
# ============================================================================

st.set_page_config(
    page_title = "Adaptive LLM Router",
    page_icon  = "⚡",
    layout     = "wide",
)


# ============================================================================
# Constants
# ============================================================================

BASELINE_AVG   = 69.03
BASELINE_MODEL = "llama-3.3-70b-versatile"


def get_weight(model):
    if "8b" in model.lower() or "instant" in model.lower():
        return 1.7
    if "70b" in model.lower() or "versatile" in model.lower():
        return 8.0
    return 0.0


# ============================================================================
# Session state
# ============================================================================

if "history" not in st.session_state:
    st.session_state.history            = []
if "total_full_s" not in st.session_state:
    st.session_state.total_full_s       = 0.0
if "total_compute_full" not in st.session_state:
    st.session_state.total_compute_full = 0.0
if "total_compute_base" not in st.session_state:
    st.session_state.total_compute_base = 0.0


# ============================================================================
# Load pipeline
# ============================================================================

@st.cache_resource
def load_pipeline():
    return Pipeline(use_cache=True, use_compression=True)


# ============================================================================
# Check API key
# ============================================================================

if not os.environ.get("GROQ_API_KEY"):
    st.error(
        "⚠️ GROQ_API_KEY not set. "
        "Add it in HuggingFace Space Settings → Variables and Secrets."
    )
    st.stop()


# ============================================================================
# Header
# ============================================================================

st.title("⚡ Adaptive LLM Routing System")
st.markdown(
    "Final Year AI Project — Routes queries to the right model automatically. "
    "**Simple queries** → fast 8B model. "
    "**Complex queries** → compressed by 8B → answered by 70B. "
    "**Repeated queries** → semantic cache (near-zero latency)."
)
st.divider()


# ============================================================================
# Sidebar
# ============================================================================

with st.sidebar:
    st.header("📊 Session Statistics")

    n             = len(st.session_state.history)
    cache_hits    = sum(1 for h in st.session_state.history if h["cache_hit"])
    simple_count  = sum(1 for h in st.session_state.history if h["complexity"] == "SIMPLE")
    complex_count = sum(1 for h in st.session_state.history if h["complexity"] == "COMPLEX")
    compressed    = sum(1 for h in st.session_state.history if h["compressed_query"])

    st.metric("Total Queries",      n)
    st.metric("Cache Hits",         cache_hits,
              delta=f"{cache_hits/n*100:.0f}%" if n else None)
    st.metric("Simple → 8B fast",  simple_count)
    st.metric("Complex → 70B",     complex_count)
    st.metric("Compressed",        compressed)

    st.divider()
    st.subheader("⏱ Latency")
    if n > 0:
        avg  = st.session_state.total_full_s / n
        pct  = (1 - avg / BASELINE_AVG) * 100
        st.metric("Avg Latency", f"{avg:.1f}s",
                  delta=f"{pct:.1f}% vs baseline",
                  delta_color="normal" if pct > 0 else "inverse")
    st.metric("Baseline", f"{BASELINE_AVG:.1f}s")

    st.divider()
    st.subheader("💰 Compute")
    if st.session_state.total_compute_base > 0:
        red = (1 - st.session_state.total_compute_full /
               st.session_state.total_compute_base) * 100
        st.metric("Compute Reduction", f"{red:.1f}%", delta="vs always 70B")

    if st.button("🗑 Clear History", use_container_width=True):
        st.session_state.history            = []
        st.session_state.total_full_s       = 0.0
        st.session_state.total_compute_full = 0.0
        st.session_state.total_compute_base = 0.0
        st.rerun()

    if st.button("🧹 Clear Cache", use_container_width=True):
        load_pipeline().cache.clear()
        st.success("Cache cleared.")


# ============================================================================
# Input
# ============================================================================

col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input(
        "Enter your query:",
        placeholder="e.g. What is the capital of France? / Explain how gold prices work",
    )
with col2:
    st.markdown("<br>", unsafe_allow_html=True)
    submit = st.button("🚀 Submit", use_container_width=True)

st.markdown("**Try these:**")
ex_cols = st.columns(4)
examples = [
    "What is the capital of France?",
    "Who invented the telephone?",
    "Explain how gold prices are determined in global markets.",
    "What are the risks of prop trading without risk management?",
]
for col, ex in zip(ex_cols, examples):
    with col:
        label = ex[:35] + "..." if len(ex) > 35 else ex
        if st.button(label, use_container_width=True):
            query  = ex
            submit = True


# ============================================================================
# Process
# ============================================================================

if submit and query.strip():
    pipeline = load_pipeline()

    with st.spinner("Processing..."):
        result = pipeline.answer(query.strip())

    st.divider()

    if result.cache_hit:
        st.success("⚡ **CACHE HIT** — Returned instantly from semantic cache (0 tokens)")
    elif result.complexity == "SIMPLE":
        st.info("🟢 **SIMPLE** → fast 8B model (llama-3.1-8b-instant)")
    else:
        st.warning("🟠 **COMPLEX** → compressed by 8B → answered by 70B (llama-3.3-70b-versatile)")

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Total",      f"{result.total_s:.2f}s")
    m2.metric("Classify",   f"{result.classifier_s:.2f}s")
    m3.metric("Compress",   f"{result.compress_s:.2f}s")
    m4.metric("Generate",   f"{result.generation_s:.2f}s")
    m5.metric("Model",      result.model_used.split("-")[0] + "..." if len(result.model_used) > 15 else result.model_used)
    saving = max(0, BASELINE_AVG - result.total_s)
    m6.metric("Time Saved", f"{saving:.1f}s", delta=f"{saving/BASELINE_AVG*100:.0f}%")

    if result.compressed_query:
        st.subheader("🔍 Compressed Brief (sent to 70B model)")
        st.info(result.compressed_query)
        st.caption("The 8B model rewrote your query. The 70B model answered this — not your raw query.")

    st.subheader("💬 Answer")
    st.markdown(result.answer)

    toks    = (result.input_tokens or 0) + (result.output_tokens or 0)
    cu_full = 0.0 if result.cache_hit else toks * get_weight(result.model_used)
    cu_base = toks * get_weight(BASELINE_MODEL)

    st.session_state.total_full_s       += result.total_s
    st.session_state.total_compute_full += cu_full
    st.session_state.total_compute_base += cu_base

    st.session_state.history.append({
        "query"           : query.strip(),
        "complexity"      : result.complexity,
        "cache_hit"       : result.cache_hit,
        "model_used"      : result.model_used,
        "total_s"         : result.total_s,
        "classifier_s"    : result.classifier_s,
        "compress_s"      : result.compress_s,
        "generation_s"    : result.generation_s,
        "compressed_query": result.compressed_query,
        "answer"          : result.answer[:300],
        "compute_units"   : cu_full,
    })


# ============================================================================
# History
# ============================================================================

if st.session_state.history:
    st.divider()
    st.subheader("📋 Query History")

    for i, h in enumerate(reversed(st.session_state.history), 1):
        icon = "⚡" if h["cache_hit"] else ("🟢" if h["complexity"] == "SIMPLE" else "🟠")

        with st.expander(
            f"{icon} [{i}] {h['query'][:60]} — {h['total_s']:.1f}s",
            expanded=(i == 1),
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Complexity",    h["complexity"])
            c2.metric("Cache Hit",     "Yes" if h["cache_hit"] else "No")
            c3.metric("Compress",      f"{h['compress_s']:.2f}s")
            c4.metric("Compute Units", f"{h['compute_units']:.0f}")

            if h["compressed_query"]:
                st.markdown("**Compressed brief:**")
                st.info(h["compressed_query"])

            st.markdown("**Answer:**")
            st.markdown(h["answer"] + ("..." if len(h["answer"]) == 300 else ""))
