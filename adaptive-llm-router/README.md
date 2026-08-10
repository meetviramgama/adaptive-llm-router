---
title: Adaptive LLM Router
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: 1.28.0
app_file: app.py
pinned: false
---

# Adaptive LLM Routing with Intelligent Response Caching

Final Year AI Project

## What it does

Automatically routes queries to the most appropriate LLM:

- **Simple queries** → fast 8B model (llama-3.1-8b-instant)
- **Complex queries** → compressed by 8B → answered by 70B (llama-3.3-70b-versatile)
- **Repeated queries** → semantic cache (near-zero latency)

## Key features

- 92.3% classifier accuracy
- 50% of queries avoid expensive model
- 10.7% compute cost reduction
- Semantic caching with paraphrase detection
- Query compression for complex queries

## Setup

Add `GROQ_API_KEY` in Space Settings → Variables and Secrets.
