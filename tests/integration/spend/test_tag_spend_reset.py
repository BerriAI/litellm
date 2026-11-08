import json
import uuid
from typing import Final

import httpx

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.database import read_rows
from tests.integration._support.provider import PROVIDER_URL, SharedProvider
from tests.integration._support.wire import Reply


def test_admin_can_reset_tag_spend_and_resume_budget_blocked_requests(
    gateway: Gateway, provider: SharedProvider
) -> None:
    tag: Final = f"tag-reset-{uuid.uuid4().hex}"
    completion: Final = Reply(
        body=json.dumps(
            {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            }
        ).encode()
    )
    with gateway.scenario() as scenario:
        model: Final = scenario.model(
            api_base=f"{PROVIDER_URL}/v1", input_cost_per_token=0.001, output_cost_per_token=0.002
        )
        gateway.post("/tag/new", {"name": tag, "max_budget": 0.0001, "budget_duration": "30d"})

        def delete_tag() -> None:
            gateway.post("/tag/delete", {"name": tag})

        scenario.cleanups.callback(delete_tag)

        def tagged_request() -> httpx.Response:
            return gateway.request(
                "POST",
                "/v1/chat/completions",
                {
                    "model": model,
                    "messages": [{"role": "user", "content": f"tag reset {uuid.uuid4().hex}"}],
                    "metadata": {"tags": [tag]},
                },
            )

        provider.expect(completion)
        first: Final = tagged_request()
        assert first.status_code == 200, first.text
        assert len(provider.received()) == 1
        eventually(
            lambda: read_rows('SELECT spend FROM "LiteLLM_TagTable" WHERE tag_name=%s', (tag,)),
            lambda rows: len(rows) == 1 and float(str(rows[0]["spend"])) > 0.0001,
            seconds=70,
        )
        blocked: Final = tagged_request()
        assert blocked.status_code == 422, blocked.text
        assert blocked.json()["error"]["type"] == "budget_exceeded", blocked.text
        assert f"Tag={tag}" in blocked.text, blocked.text
        assert provider.received() == ()
        reset: Final = gateway.request("POST", "/tag/update", {"name": tag, "spend": 0})
        assert reset.status_code == 200, reset.text
        provider.expect(completion)
        try:
            resumed: Final = tagged_request()
            assert resumed.status_code == 200, resumed.text
            assert len(provider.received()) == 1
        finally:
            provider.replies.clear()
