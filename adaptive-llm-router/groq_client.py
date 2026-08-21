"""
Groq API Client
---------------
Cloud LLM client for Adaptive LLM Routing System.

Models:
    Low tier  : openai/gpt-oss-20b
    High tier : openai/gpt-oss-120b
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

from groq import Groq


# ============================================================================
# MODELS
# ============================================================================

LOW_MODEL = "openai/gpt-oss-20b"
HIGH_MODEL = "openai/gpt-oss-120b"


# ============================================================================
# RESPONSE
# ============================================================================

@dataclass
class GroqResponse:
    text: str
    model: str
    wall_clock_s: float
    prompt_eval_count: int
    eval_count: int
    tokens_per_second: float


# ============================================================================
# CLIENT
# ============================================================================

_client = None


def get_client() -> Groq:
    global _client

    if _client is None:

        api_key = os.environ.get("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. "
                "Add it in Hugging Face Spaces → Settings → Secrets."
            )

        _client = Groq(api_key=api_key)

    return _client


# ============================================================================
# GENERATE
# ============================================================================

def generate(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.7,
    think: bool = True,
) -> GroqResponse:

    client = get_client()

    messages = []

    if system:
        messages.append(
            {
                "role": "system",
                "content": system,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    start = time.perf_counter()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    wall_clock = time.perf_counter() - start

    text = ""

    if response.choices:
        text = response.choices[0].message.content or ""

    input_tokens = 0
    output_tokens = 0

    if response.usage:

        input_tokens = (
            getattr(response.usage, "prompt_tokens", 0)
            or 0
        )

        output_tokens = (
            getattr(response.usage, "completion_tokens", 0)
            or 0
        )

    tokens_per_second = (
        output_tokens / wall_clock
        if wall_clock > 0
        else 0.0
    )

    return GroqResponse(
        text=text.strip(),
        model=model,
        wall_clock_s=wall_clock,
        prompt_eval_count=input_tokens,
        eval_count=output_tokens,
        tokens_per_second=tokens_per_second,
    )


# ============================================================================
# TEST
# ============================================================================

if __name__ == "__main__":

    print("Testing Groq client...")

    for model, label in [
        (LOW_MODEL, "GPT-OSS-20B"),
        (HIGH_MODEL, "GPT-OSS-120B"),
    ]:

        print()
        print("=" * 60)
        print(label)
        print("=" * 60)

        try:

            result = generate(
                model=model,
                prompt="What is the capital of France? Answer in one word.",
                max_tokens=20,
                temperature=0.0,
            )

            print("Response:", result.text)
            print(f"Latency: {result.wall_clock_s:.2f}s")
            print(
                f"Tokens: "
                f"{result.prompt_eval_count} input / "
                f"{result.eval_count} output"
            )
            print(
                f"Speed: "
                f"{result.tokens_per_second:.1f} tok/s"
            )

        except Exception as e:

            print("ERROR:", e)