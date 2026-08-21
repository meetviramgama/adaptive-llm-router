"""
Groq API Client
---------------
Cloud LLM client for the Adaptive LLM Routing System.

Models:
    Low tier  : openai/gpt-oss-20b
    High tier : openai/gpt-oss-120b

Usage:
    Set GROQ_API_KEY environment variable before running.

    HuggingFace Spaces:
        Settings → Secrets → GROQ_API_KEY

    Never hardcode the API key in code.
"""

import os
import time
from dataclasses import dataclass
from typing import Optional

from groq import Groq


# ============================================================================
# Models
# ============================================================================

LOW_MODEL = "openai/gpt-oss-20b"
HIGH_MODEL = "openai/gpt-oss-120b"


# ============================================================================
# Response dataclass
# Same interface as the previous OllamaResponse
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
# Client
# ============================================================================

def get_client() -> Groq:
    """
    Create and return a Groq client using GROQ_API_KEY.
    """

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY environment variable not set. "
            "Set it in HuggingFace Spaces Secrets or your .env file."
        )

    return Groq(api_key=api_key)


# ============================================================================
# Generate
# ============================================================================

def generate(
    model: str,
    prompt: str,
    system: Optional[str] = None,
    max_tokens: int = 500,
    temperature: float = 0.7,
    think: bool = True,
) -> GroqResponse:
    """
    Call Groq API and return response text + timing/token metrics.

    Parameters:
        model:
            LOW_MODEL or HIGH_MODEL

        prompt:
            User/query prompt

        system:
            Optional system instruction

        max_tokens:
            Maximum output tokens

        temperature:
            Sampling temperature

        think:
            Kept for compatibility with previous Ollama interface.
            Currently ignored.
    """

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

    start = time.time()

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    wall_clock = time.time() - start

    text = response.choices[0].message.content or ""

    input_tokens = 0
    output_tokens = 0

    if response.usage:
        input_tokens = response.usage.prompt_tokens or 0
        output_tokens = response.usage.completion_tokens or 0

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
# Local test
# ============================================================================

if __name__ == "__main__":

    print("Testing Groq client...")

    for model, label in [
        (LOW_MODEL, "Low Tier - GPT-OSS-20B"),
        (HIGH_MODEL, "High Tier - GPT-OSS-120B"),
    ]:

        print(f"\n--- {label} ---")
        print(f"Model: {model}")

        try:
            result = generate(
                model=model,
                prompt="What is the capital of France? Answer in one word.",
                max_tokens=20,
                temperature=0.0,
            )

            print(f"Response:   {result.text}")
            print(f"Latency:    {result.wall_clock_s:.2f}s")
            print(
                f"Tokens:     "
                f"{result.prompt_eval_count} in / "
                f"{result.eval_count} out"
            )
            print(
                f"Speed:      "
                f"{result.tokens_per_second:.1f} tok/s"
            )

        except Exception as e:
            print(f"ERROR: {e}")