"""
Groq API Client
---------------
Replaces ollama_client.py for cloud deployment.
Uses Groq's API instead of local Ollama server.

Models:
    Low tier:  llama-3.1-8b-instant     (replaces qwen3:1.7b)
    High tier: llama-3.3-70b-versatile  (replaces qwen3:8b)

Usage:
    Set GROQ_API_KEY environment variable before running.
    Never hardcode the key in code.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional
from groq import Groq


# ============================================================================
# Models
# ============================================================================

LOW_MODEL  = "openai/gpt-oss-20b"
HIGH_MODEL = "openai/gpt-oss-120b"

# ============================================================================
# Response dataclass — same interface as OllamaResponse
# ============================================================================

@dataclass
class GroqResponse:
    text: str
    model: str
    wall_clock_s: float
    prompt_eval_count: int    # input tokens
    eval_count: int           # output tokens
    tokens_per_second: float


# ============================================================================
# Client
# ============================================================================

def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Set it in HuggingFace Spaces secrets or your .env file."
        )
    return Groq(api_key=api_key)


def generate(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.7,
    think: bool = True,       # kept for interface compatibility — ignored on Groq
) -> GroqResponse:
    """
    Call Groq API and return text + timing metrics.
    Same interface as ollama_client.generate() for drop-in replacement.
    """
    client = get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    start = time.time()
    response = client.chat.completions.create(
        model       = model,
        messages    = messages,
        max_tokens  = max_tokens,
        temperature = temperature,
    )
    wall_clock = time.time() - start

    text         = response.choices[0].message.content or ""
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    tps           = output_tokens / wall_clock if wall_clock > 0 else 0.0

    return GroqResponse(
        text               = text.strip(),
        model              = model,
        wall_clock_s       = wall_clock,
        prompt_eval_count  = input_tokens,
        eval_count         = output_tokens,
        tokens_per_second  = tps,
    )


if __name__ == "__main__":
    print("Testing Groq client...")
    for model, label in [(LOW_MODEL, "Low tier"), (HIGH_MODEL, "High tier")]:
        print(f"\n--- {label}: {model} ---")
        result = generate(model, "What is the capital of France? Answer in one word.")
        print(f"Response:   {result.text}")
        print(f"Latency:    {result.wall_clock_s:.2f}s")
        print(f"Tokens:     {result.prompt_eval_count} in / {result.eval_count} out")
        print(f"Speed:      {result.tokens_per_second:.1f} tok/s")
