import json
import uuid
from typing import Final

import pytest
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("spend.failed_dispatch.failure_row_records_estimated_input_tokens")
def test_provider_500_after_dispatch_records_estimated_prompt_tokens_on_failure_row(gateway: Gateway) -> None:
    prompt: Final = "failed dispatch accounting " + uuid.uuid4().hex
    system: Final = "You are a terse accounting assistant"

    def provider(request: Request) -> Reply:
        assert request.method == "POST" and request.target == "/v1/chat/completions"
        body: Final = json.loads(request.body)
        assert body["model"] == "gpt-4o-mini"
        assert body["messages"] == [{"role": "system", "content": system}, {"role": "user", "content": prompt}]
        return Reply(
            status=500,
            body=b'{"error":{"message":"synthetic provider outage","type":"server_error","code":"500"}}',
        )

    with wire_server(provider) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=wire.url + "/v1", num_retries=0)
        key: Final = scenario.key(models=[model])
        failed: Final = gateway.request(
            "POST",
            "/v1/chat/completions",
            {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}]},
            key=key,
        )
        assert failed.status_code == 500 and "synthetic provider outage" in failed.text, failed.text
        call_id: Final = failed.headers["x-litellm-call-id"]
        assert len(wire.drain()) == 1
        rows: Final = eventually(
            lambda: read_rows(
                'SELECT status, spend, prompt_tokens, completion_tokens, total_tokens FROM "LiteLLM_SpendLogs" '
                "WHERE request_id=%s",
                (call_id,),
            ),
            lambda values: len(values) == 1,
            seconds=70,
        )
        row: Final = rows[0]
        assert row["status"] == "failure" and float(row["spend"]) == 0 and row["completion_tokens"] == 0, row
        assert row["prompt_tokens"] > 0, f"failure row lost the dispatched input tokens: {row}"
        assert row["total_tokens"] == row["prompt_tokens"], row
