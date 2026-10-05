import json
import time
import uuid
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import httpx
import pytest
from integration._support.client import Gateway
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

KEY_TPM_LIMIT: Final = 100
MAX_TOKENS: Final = 80
CONCURRENT_REQUESTS: Final = 10
PROVIDER_HOLD_SECONDS: Final = 2.0
UPSTREAM_REPLY: Final = json.dumps(
    {
        "id": "chatcmpl_tpm_reservation",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-4o-mini",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "reserved"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40},
    }
).encode()


@pytest.mark.covers("quota_management.key_tpm_limit.concurrent_requests_reserve_tokens_before_provider_call")
def test_concurrent_requests_over_key_tpm_are_rejected_before_reaching_provider(gateway: Gateway) -> None:
    probe: Final = "tpm reservation probe " + uuid.uuid4().hex[:8]
    messages: Final[list[JsonValue]] = [{"role": "user", "content": probe}]

    def respond(request: Request) -> Reply:
        assert (request.method, request.target) == ("POST", "/v1/chat/completions")
        assert json.loads(request.body) == {"model": "gpt-4o-mini", "max_tokens": MAX_TOKENS, "messages": messages}
        time.sleep(PROVIDER_HOLD_SECONDS)
        return Reply(body=UPSTREAM_REPLY)

    with wire_server(respond) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(tpm_limit=KEY_TPM_LIMIT)
        body: Final[dict[str, JsonValue]] = {
            "model": model,
            "max_tokens": MAX_TOKENS,
            "messages": messages,
        }

        def send(_: int) -> httpx.Response:
            return gateway.request("POST", "/v1/chat/completions", body, key=key)

        with ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as pool:
            responses: Final = tuple(pool.map(send, range(CONCURRENT_REQUESTS)))
        statuses: Final = Counter(response.status_code for response in responses)
        assert statuses == Counter({200: 1, 429: CONCURRENT_REQUESTS - 1}), tuple(
            response.text for response in responses
        )
        assert tuple(json.loads(request.body)["messages"] for request in wire.drain()) == (messages,)
