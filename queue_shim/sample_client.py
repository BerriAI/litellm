"""Point your agent at the shim instead of litellm.

Run this against a live shim and a healthy-or-unhealthy litellm. The first
request holds while the queue is drained, the second arrives after litellm
is back and is forwarded immediately.
"""

import httpx

from queue_shim.router import COMPLETIONS_ENDPOINT

SHIM_BASE = "http://localhost:4001"


def main() -> None:
    body = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}]}
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(f"{SHIM_BASE}/v1/chat/completions", json=body)
        print("status:", resp.status_code)
        print(resp.json())

        resp2 = client.post(f"{SHIM_BASE}/v1/chat/completions", json=body)
        print("retry status:", resp2.status_code)
        print(resp2.json())


if __name__ == "__main__":
    main()
