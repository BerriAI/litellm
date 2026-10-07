import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import pytest
import yaml

from integration._support.client import Gateway, eventually
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server


@pytest.mark.covers(
    "other.observability.callbacks.credentials_stay_out_of_event_bodies",
    "other.observability.callbacks.concurrent_results_join_complete_events_and_rows",
)
def test_concurrent_success_and_failure_join_callbacks_and_rows_without_credentials(
    gateway: Gateway, tmp_path: Path
) -> None:
    marker: Final = "callback" + uuid.uuid4().hex
    secret: Final = "synthetic-provider-secret-" + marker
    sink_secret: Final = "synthetic-sink-secret-" + marker

    def upstream(request: Request) -> Reply:
        body: Final = json.loads(request.body)
        text: Final = body["messages"][0]["content"]
        assert request.headers["authorization"] == f"Bearer {secret}"
        if text.endswith("failure"):
            return Reply(
                status=400,
                body=json.dumps(
                    {
                        "error": {
                            "type": "invalid_request_error",
                            "code": "synthetic_failure",
                            "message": "synthetic callback failure",
                        }
                    }
                ).encode(),
            )
        return Reply(
            body=json.dumps(
                {
                    "id": text,
                    "object": "chat.completion",
                    "created": 1,
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
                }
            ).encode()
        )

    def sink(request: Request) -> Reply:
        assert request.headers["authorization"] == f"Bearer {sink_secret}"
        return Reply()

    with wire_server(upstream) as provider, wire_server(sink) as endpoint:
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["litellm_settings"].update({"callbacks": ["generic_api"], "DEFAULT_FLUSH_INTERVAL_SECONDS": 1})
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
            tags: Final = tuple(f"{marker}-{index}-{'failure' if index % 2 else 'success'}" for index in range(4))

            def request(tag: str):
                return candidate.request(
                    "POST",
                    "/v1/chat/completions",
                    {
                        "model": model,
                        "messages": [{"role": "user", "content": tag}],
                        "metadata": {"tags": [tag]},
                        "cache": {"no-cache": True},
                    },
                    key=key,
                )

            with ThreadPoolExecutor(max_workers=4) as pool:
                responses: Final = tuple(pool.map(request, tags))
            assert tuple(response.status_code for response in responses) == (200, 400, 200, 400)
            assert len(provider.drain()) == 4
            batches = []

            def delivered() -> tuple[dict, ...]:
                batches.extend(endpoint.drain())
                return tuple(
                    event
                    for batch in batches
                    for event in json.loads(batch.body)
                    if any(tag in event.get("request_tags", []) for tag in tags)
                )

            events: Final = eventually(delivered, lambda values: len(values) == 4, seconds=10)
            body: Final = b"".join(batch.body for batch in batches)
            for credential in (secret, sink_secret, key, candidate.key):
                assert credential.encode() not in body
            assert len({event["id"] for event in events}) == 4
            assert {tuple(tag for tag in event["request_tags"] if tag in tags) for event in events} == {
                (tag,) for tag in tags
            }
            for tag, response in zip(tags, responses, strict=True):
                event: Final = next(event for event in events if tag in event["request_tags"])
                assert event["litellm_call_id"] == response.headers["x-litellm-call-id"]
                assert event["status"] == ("failure" if tag.endswith("failure") else "success")
                if response.status_code == 200:
                    assert response.json()["id"] == event["id"] == tag
                    assert response.json()["choices"][0]["message"]["content"] == tag
                    assert event["prompt_tokens"] == 11 and event["completion_tokens"] == 4
                    assert event["response_cost"] == pytest.approx(0.019)
                else:
                    assert event["response_cost"] == 0
                    assert "synthetic callback failure" in json.dumps(event["error_information"])
                rows: Final = eventually(
                    lambda identity=event["id"]: read_rows(
                        'SELECT request_id, spend, prompt_tokens, completion_tokens, request_tags '
                        'FROM "LiteLLM_SpendLogs" WHERE request_id=%s',
                        (identity,),
                    ),
                    lambda values: len(values) == 1,
                    seconds=70,
                )
                saved_tags: Final = (
                    json.loads(rows[0]["request_tags"])
                    if isinstance(rows[0]["request_tags"], str)
                    else rows[0]["request_tags"]
                )
                assert [value for value in saved_tags if value in tags] == [tag]
                assert float(rows[0]["spend"]) == pytest.approx(event["response_cost"])
                assert rows[0]["completion_tokens"] == event["completion_tokens"]
                if response.status_code == 200:
                    assert rows[0]["prompt_tokens"] == event["prompt_tokens"]
                else:
                    assert event["prompt_tokens"] == event["completion_tokens"] == rows[0]["completion_tokens"] == 0
