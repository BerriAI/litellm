import json
import uuid
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers("other.observability.callbacks.responses_success_cost_matches_spend_row")
def test_responses_success_callback_cost_is_priced_and_matches_spend_row(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "responses-cost-" + uuid.uuid4().hex
    secret: Final = "synthetic-provider-secret-" + marker
    sink_secret: Final = "synthetic-sink-secret-" + marker

    def upstream(request: Request) -> Reply:
        assert request.target == "/v1/responses"
        assert request.headers["authorization"] == f"Bearer {secret}"
        body: Final = json.loads(request.body)
        assert body["input"] == marker
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_" + marker,
                    "object": "response",
                    "created_at": 1,
                    "status": "completed",
                    "model": "gpt-4o-mini",
                    "output": [
                        {
                            "id": "msg_" + marker,
                            "type": "message",
                            "role": "assistant",
                            "status": "completed",
                            "content": [{"type": "output_text", "text": marker, "annotations": []}],
                        }
                    ],
                    "usage": {
                        "input_tokens": 11,
                        "output_tokens": 4,
                        "total_tokens": 15,
                        "input_tokens_details": {"cached_tokens": 0},
                        "output_tokens_details": {"reasoning_tokens": 0},
                    },
                    "parallel_tool_calls": True,
                    "tool_choice": "auto",
                    "tools": [],
                }
            ).encode()
        )

    def sink(request: Request) -> Reply:
        assert request.headers["authorization"] == f"Bearer {sink_secret}"
        return Reply()

    with wire_server(upstream) as provider, wire_server(sink) as endpoint:
        base: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config: Final = {
            **base,
            "litellm_settings": {
                **base["litellm_settings"],
                "callbacks": ["generic_api"],
                "DEFAULT_FLUSH_INTERVAL_SECONDS": 1,
            },
        }
        path: Final = tmp_path / "callbacks.yaml"
        path.write_text(yaml.safe_dump(config))
        with (
            owned_proxy(
                gateway,
                tmp_path,
                {
                    "GENERIC_LOGGER_ENDPOINT": endpoint.url,
                    "GENERIC_LOGGER_HEADERS": f"Authorization=Bearer {sink_secret}",
                },
                config=path,
            ) as candidate,
            candidate.scenario() as scenario,
        ):
            model: Final = scenario.model(
                api_base=provider.url + "/v1", api_key=secret, input_cost_per_token=0.001, output_cost_per_token=0.002
            )
            key: Final = scenario.key(models=[model])
            response: Final = candidate.request(
                "POST",
                "/v1/responses",
                {"model": model, "input": marker, "metadata": {"tags": [marker]}},
                key=key,
            )
            assert response.status_code == 200, response.text
            assert len(provider.drain()) == 1
            response_id: Final = response.json()["id"]

            def delivered() -> tuple[dict, ...]:
                return tuple(
                    event
                    for batch in endpoint.drain()
                    for event in json.loads(batch.body)
                    if marker in event.get("request_tags", [])
                )

            events: Final = eventually(delivered, lambda values: len(values) == 1, seconds=10)
            event: Final = events[0]
            assert event["status"] == "success"
            assert event["call_type"] in ("responses", "aresponses")
            assert event["prompt_tokens"] == 11 and event["completion_tokens"] == 4
            assert event["response_cost"] == pytest.approx(11 * 0.001 + 4 * 0.002)
            rows: Final = eventually(
                lambda: read_rows(
                    'SELECT spend, call_type FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                    (response_id,),
                ),
                lambda values: len(values) == 1,
                seconds=70,
            )
            assert float(rows[0]["spend"]) == pytest.approx(event["response_cost"])
            assert rows[0]["call_type"] == event["call_type"]
