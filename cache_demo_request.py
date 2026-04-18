"""
Demo script: back-to-back Bedrock streaming requests with prompt caching.
Request 1: populates the cache (cache_creation_input_tokens)
Request 2: reads from cache (cache_read_input_tokens)
"""

import json
import time

import httpx

PROXY_URL = "http://localhost:4001"
API_KEY = "sk-1234"

# ~5000 token system prompt (above claude-haiku-4-5's 2048-token min for caching on Bedrock)
LARGE_SYSTEM_PROMPT = (
    "AWS Bedrock provides managed ML infrastructure for enterprise workloads. "
    "Anthropic Claude models support prompt caching for cost optimization. "
) * 200


def make_streaming_request(req_num: int, label: str) -> None:
    print(f"\n{'='*60}")
    print(f"Request {req_num}: {label}")
    print(f"{'='*60}")

    payload = {
        "model": "bedrock-claude-haiku",
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": LARGE_SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {"role": "user", "content": f"Say only: 'Request {req_num} done'"},
        ],
        "stream": True,
        "max_tokens": 20,
    }

    full_response = ""
    with httpx.Client(timeout=60) as client:
        with client.stream(
            "POST",
            f"{PROXY_URL}/v1/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}"},
        ) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    if content := delta.get("content"):
                        full_response += content

    print(f"Response: {full_response!r}")


if __name__ == "__main__":
    print("Sending request 1 (cache write)...")
    make_streaming_request(1, "cache WRITE (populates cache)")

    print("\nWaiting 2s between requests...")
    time.sleep(2)

    print("Sending request 2 (cache read)...")
    make_streaming_request(2, "cache READ (hits cache)")

    print("\n\nDone. Check SpendLogs in the DB or the UI at http://localhost:4001")
