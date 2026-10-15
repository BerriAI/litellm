import base64
import json
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, JsonValue, eventually, object_value, string_value
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
            batches: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls must keep earlier batches

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
                        "SELECT request_id, spend, prompt_tokens, completion_tokens, request_tags "
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


def _responses_frames(identity: str, text: str) -> tuple[bytes, ...]:
    output: Final = [
        {
            "type": "message",
            "id": f"msg_{identity}",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": text, "annotations": []}],
        }
    ]
    completed: Final = {
        "id": identity,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-4o-mini",
        "output": output,
        "usage": {
            "input_tokens": 11,
            "output_tokens": 4,
            "total_tokens": 15,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    events: Final = (
        {"type": "response.created", "response": {**completed, "status": "in_progress", "output": [], "usage": None}},
        {
            "type": "response.output_text.delta",
            "item_id": f"msg_{identity}",
            "output_index": 0,
            "content_index": 0,
            "delta": text,
        },
        {"type": "response.completed", "response": completed},
    )
    return tuple(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n".encode() for event in events)


@pytest.mark.covers("other.observability.callbacks.streamed_responses_events_carry_provider_response_headers")
def test_streamed_responses_success_callback_carries_provider_apim_request_id(gateway: Gateway, tmp_path: Path) -> None:
    marker: Final = "resp_" + uuid.uuid4().hex
    correlation: Final = "azure-correlation-" + marker
    region: Final = "East US 2"
    secret: Final = "synthetic-provider-secret-" + marker
    sink_secret: Final = "synthetic-sink-secret-" + marker

    def upstream(request: Request) -> Reply:
        assert request.target.endswith("/responses"), request.target
        assert request.headers["authorization"] == f"Bearer {secret}"
        assert json.loads(request.body) == {
            "model": "gpt-4o-mini",
            "input": "header control " + marker,
            "stream": True,
        }, request.body
        return Reply(
            content_type="text/event-stream",
            chunks=_responses_frames(marker, "streamed control"),
            headers={"apim-request-id": correlation, "x-ms-region": region},
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
            model: Final = scenario.model(api_base=provider.url + "/v1", api_key=secret)
            response: Final = candidate.request(
                "POST", "/v1/responses", {"model": model, "input": "header control " + marker, "stream": True}
            )
            assert response.status_code == 200, response.text
            assert f'"item_id":"msg_{marker}"' in response.text, response.text
            assert '"type":"response.completed"' in response.text, response.text
            assert len(provider.drain()) == 1
            batches: Final[list[Request]] = []  # mutable-ok: drain() consumes the queue, later polls must keep earlier batches

            def delivered() -> tuple[dict, ...]:
                batches.extend(endpoint.drain())
                return tuple(
                    event for batch in batches for event in json.loads(batch.body) if event.get("model_group") == model
                )

            events: Final = eventually(delivered, lambda values: len(values) == 1, seconds=10)
            assert (events[0]["status"], events[0]["stream"], events[0]["call_type"]) == ("success", True, "aresponses")
            additional_headers: Final = events[0]["hidden_params"]["additional_headers"] or {}
            provider_headers: Final = {
                name: value
                for name, value in additional_headers.items()
                if name in ("llm_provider-apim-request-id", "llm_provider-x-ms-region")
            }
            assert provider_headers == {
                "llm_provider-apim-request-id": correlation,
                "llm_provider-x-ms-region": region,
            }, json.dumps(events[0]["hidden_params"])


_RAISING_HOOK: Final = """
from litellm.integrations.custom_logger import CustomLogger


class RaisingHook(CustomLogger):
    async def async_post_call_success_deployment_hook(self, request_data, response, call_type):
        raise RuntimeError(f"hook rejected {type(response).__name__} for {call_type}")


instance = RaisingHook()
"""

_VIDEO_JOB: Final = {
    "id": "video_hook_isolation",
    "object": "video",
    "status": "queued",
    "model": "sora-2",
    "seconds": "4",
    "size": "720x1280",
}

_UPSTREAM_REPLIES: Final[Mapping[str, Mapping[str, JsonValue]]] = {
    "/v1/chat/completions": {
        "id": "chatcmpl_hook_isolation",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.6",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    },
    "/v1/embeddings": {
        "object": "list",
        "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
        "model": "text-embedding-3-small",
        "usage": {"prompt_tokens": 1, "total_tokens": 1},
    },
    "/v1/responses": {
        "id": "resp_hook_isolation",
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": "gpt-5.6",
        "output": [
            {
                "type": "message",
                "id": "msg_hook_isolation",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hi", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
    },
    "/v1/videos": _VIDEO_JOB,
}


def _item(value: JsonValue, index: int) -> JsonValue:
    assert isinstance(value, list), f"Expected a list, received {type(value).__name__}"
    return value[index]


def _chat_text(body: dict[str, JsonValue]) -> str:
    return string_value(object_value(object_value(_item(body["choices"], 0))["message"])["content"])


def _embedding_vector(body: dict[str, JsonValue]) -> JsonValue:
    return object_value(_item(body["data"], 0))["embedding"]


def _responses_text(body: dict[str, JsonValue]) -> str:
    return string_value(object_value(_item(object_value(_item(body["output"], 0))["content"], 0))["text"])


def _video_job(body: dict[str, JsonValue]) -> tuple[str, str]:
    encoded_id: Final = string_value(body["id"]).removeprefix("video_")
    decoded: Final = base64.b64decode(encoded_id).decode()
    return decoded.rsplit("video_id:", 1)[-1], string_value(body["status"])


@dataclass(frozen=True, slots=True)
class _Surface:
    route: str
    upstream_model: str
    body: Callable[[str], dict[str, JsonValue]]
    observed: Callable[[dict[str, JsonValue]], JsonValue | tuple[str, str]]
    expected: JsonValue | tuple[str, str]


_SURFACES: Final = (
    pytest.param(
        _Surface(
            "/v1/chat/completions",
            "openai/gpt-5.6",
            lambda model: {"model": model, "messages": [{"role": "user", "content": "hook isolation"}]},
            _chat_text,
            "hi",
        ),
        id="chat",
    ),
    pytest.param(
        _Surface(
            "/v1/embeddings",
            "openai/text-embedding-3-small",
            lambda model: {"model": model, "input": "hook isolation"},
            _embedding_vector,
            [0.1, 0.2],
        ),
        id="embeddings",
    ),
    pytest.param(
        _Surface(
            "/v1/responses",
            "openai/gpt-5.6",
            lambda model: {"model": model, "input": "hook isolation"},
            _responses_text,
            "hi",
        ),
        id="responses",
    ),
    pytest.param(
        _Surface(
            "/v1/videos",
            "openai/sora-2",
            lambda model: {"model": model, "prompt": "a cat"},
            _video_job,
            (_VIDEO_JOB["id"], _VIDEO_JOB["status"]),
        ),
        id="videos",
    ),
)


@pytest.mark.covers("other.observability.callbacks.raising_success_deployment_hook_keeps_response")
@pytest.mark.parametrize("surface", _SURFACES)
def test_response_survives_raising_success_deployment_hook(gateway: Gateway, tmp_path: Path, surface: _Surface) -> None:
    def upstream(request: Request) -> Reply:
        assert request.target == surface.route, request.target
        assert b"hook isolation" in request.body or b"a cat" in request.body, request.body[:300]
        return Reply(body=json.dumps(_UPSTREAM_REPLIES[surface.route]).encode())

    (tmp_path / "raising_hook.py").write_text(_RAISING_HOOK)
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["litellm_settings"].update({"callbacks": ["raising_hook.instance"]})
    path: Final = tmp_path / "hook.yaml"
    path.write_text(yaml.safe_dump(config))
    with (
        wire_server(upstream) as provider,
        owned_proxy(gateway, tmp_path, {}, config=path) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(model=surface.upstream_model, api_base=provider.url + "/v1")
        response: Final = candidate.request("POST", surface.route, surface.body(model))
        assert response.status_code == 200, response.text
        assert surface.observed(object_value(response.json())) == surface.expected, response.text
