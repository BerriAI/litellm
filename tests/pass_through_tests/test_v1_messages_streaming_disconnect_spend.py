"""
Regression test: /v1/messages streaming interrupted mid-stream must still
produce a spend-log entry.

On v1.79.1 the proxy records spend for the partially-streamed request.
A refactor on `main` broke that path, so the same scenario now produces
zero spend-log rows.

Run against a live proxy (e.g. ``litellm --config proxy_server_config.yaml``):

    pytest tests/pass_through_tests/test_v1_messages_streaming_disconnect_spend.py -s
"""

import asyncio
import json
import uuid

import aiohttp
import pytest


BASE_URL = "http://127.0.0.1:4000" # change appropriately
ADMIN_KEY = "sk-1234"


async def _generate_key(session: aiohttp.ClientSession) -> str:
    """Create a fresh virtual key so spend is isolated."""
    url = f"{BASE_URL}/key/generate"
    headers = {"Authorization": f"Bearer {ADMIN_KEY}", "Content-Type": "application/json"}
    async with session.post(url, headers=headers, json={"models": []}) as resp:
        assert resp.status == 200, f"key/generate failed: {await resp.text()}"
        data = await resp.json()
        return data["key"]


async def _get_spend_logs_by_spend_id(session: aiohttp.ClientSession, api_key: str, spend_id: str):
    """Query /spend/logs by api_key then filter by spend_id in metadata."""
    url = f"{BASE_URL}/spend/logs?api_key={api_key}"
    headers = {"Authorization": f"Bearer {ADMIN_KEY}", "Content-Type": "application/json"}
    async with session.get(url, headers=headers) as resp:
        assert resp.status == 200, f"spend/logs failed: {await resp.text()}"
        all_logs = await resp.json()
        if not isinstance(all_logs, list):
            return []
        matched = []
        for log in all_logs:
            meta = log.get("metadata")
            if isinstance(meta, str):
                meta = json.loads(meta)
            if isinstance(meta, dict):
                slm = meta.get("spend_logs_metadata") or {}
                if slm.get("spend_id") == spend_id:
                    matched.append(log)
        return matched


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=2)
async def test_v1_messages_streaming_disconnect_has_spend_log():
    """
    1. Send a streaming POST to /v1/messages.
    2. Read a few SSE chunks, then close the connection (simulating a client
       disconnect / interruption).
    3. Wait for the proxy's async spend-tracking pipeline to flush.
    4. Assert that at least one spend-log row exists for the request.

    This PASSES on v1.79.1 and FAILS on the latest main branch.
    """
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=60)
    ) as session:
        key = await _generate_key(session)

        spend_id = str(uuid.uuid4())

        headers = {
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "x-litellm-spend-logs-metadata": '{"spend_id": "' + spend_id + '"}',
        }

        payload = {
            "model": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "max_tokens": 3000,
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Write several detailed paragraphs (at least 500 words) about the "
                        f"history of the Roman Empire. Unique id: {uuid.uuid4()}"
                    ),
                }
            ],
        }

        chunks_read = 0

        async with session.post(
            f"{BASE_URL}/v1/messages", json=payload, headers=headers
        ) as resp:
            assert resp.status == 200, f"/v1/messages failed: {await resp.text()}"

            async for raw_line in resp.content:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                chunks_read += 1
                print(f"  chunk #{chunks_read}: {line[:120]}")
                if chunks_read >= 5:
                    break

        assert chunks_read >= 3, (
            f"Expected at least 3 chunks before disconnect, got {chunks_read}"
        )

        print(
            f"\nDisconnected after {chunks_read} chunks.  "
            f"Waiting for spend pipeline to flush …"
        )

        spend_data = None
        max_retries = 4
        for attempt in range(1, max_retries + 1):
            await asyncio.sleep(10)
            print(f"  spend-log poll attempt {attempt}/{max_retries}")
            spend_data = await _get_spend_logs_by_spend_id(session, key, spend_id)
            if spend_data and len(spend_data) > 0:
                print(f"  ✓ found {len(spend_data)} spend-log row(s)")
                break
            print("  … not found yet")

        assert spend_data is not None and len(spend_data) > 0, (
            f"No spend-log entry found for spend_id={spend_id} "
            f"after streaming disconnect.  "
            f"This is the regression: interrupted /v1/messages streams must "
            f"still record spend."
        )

        log_entry = spend_data[0]
        print(
            f"\nSpend-log entry:\n{json.dumps(log_entry, indent=2, default=str)}"
        )

        prompt_tokens = log_entry.get("prompt_tokens", 0)
        completion_tokens = log_entry.get("completion_tokens", 0)
        assert prompt_tokens > 0, (
            "Spend-log row exists but has zero prompt tokens, so usage was not recorded."
        )
        assert completion_tokens >= 100, (
            f"Spend-log completion_tokens={completion_tokens} is far below the full "
            f"response Bedrock generated and billed. The interrupted stream was billed "
            f"on the few chunks the client drained, not the full upstream output. "
            f"chunks_read={chunks_read}, prompt_tokens={prompt_tokens}"
        )
