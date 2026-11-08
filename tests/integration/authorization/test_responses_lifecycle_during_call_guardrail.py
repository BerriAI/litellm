"""GET, compact, DELETE and cancel on /v1/responses answer 200 when a during_call guardrail is on by default
(LIT-9770): the hook used to raise `'acompact_responses' is not a valid CallTypes` and the proxy returned 500"""

import json
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, string_value
from integration._support.process import owned_proxy
from integration._support.provider import SharedProvider
from integration._support.wire import Reply
from pydantic import JsonValue, TypeAdapter

pytestmark: Final = pytest.mark.shares_worker_state

_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])
_MODEL: Final = "openai/gpt-5.4-mini"


def _input() -> list[JsonValue]:
    return [
        {"role": "user", "content": f"Remember the number {uuid.uuid4().hex}."},
        {"role": "assistant", "content": "Got it."},
    ]


def _response(response_id: str) -> Reply:
    body: Final[dict[str, JsonValue]] = {
        "id": response_id,
        "object": "response",
        "created_at": 1,
        "status": "completed",
        "model": _MODEL.removeprefix("openai/"),
        "output": [
            {
                "type": "message",
                "id": f"msg_{uuid.uuid4().hex}",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Got it.", "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "usage": {"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
    }
    return Reply(body=json.dumps(body).encode())


def _deleted(response_id: str) -> Reply:
    return Reply(body=json.dumps({"id": response_id, "object": "response.deleted", "deleted": True}).encode())


def _guarded_config(path: Path) -> Path:
    base: Final = _JSON_OBJECT.validate_python(yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text()))
    guardrail: Final[dict[str, JsonValue]] = {
        "guardrail_name": f"content-filter-{uuid.uuid4().hex}",
        "litellm_params": {
            "guardrail": "litellm_content_filter",
            "mode": "during_call",
            "default_on": True,
            "blocked_words": [{"keyword": "forbiddenword", "action": "BLOCK"}],
        },
    }
    path.write_text(yaml.safe_dump({**base, "guardrails": [guardrail]}))
    return path


@pytest.fixture(scope="module")
def guarded(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[Gateway]:
    directory: Final = tmp_path_factory.mktemp("during-call-guardrail")
    with owned_proxy(session_gateway, directory, {}, config=_guarded_config(directory / "config.yaml")) as candidate:
        yield candidate


def _create(guarded: Gateway, provider: SharedProvider, upstream_id: str) -> str:
    provider.expect(_response(upstream_id))
    created: Final = guarded.request("POST", "/v1/responses", {"model": _MODEL, "input": _input(), "store": True})
    assert created.status_code == 200, created.text
    (upstream,) = provider.received()
    assert upstream.target.endswith("/responses"), upstream.target
    return string_value(_JSON_OBJECT.validate_json(created.content)["id"])


def _lifecycle(guarded: Gateway, provider: SharedProvider, operation: str) -> tuple[httpx.Response, str]:
    if operation == "compact":
        provider.expect(_response(f"resp_{uuid.uuid4().hex}"))
        return guarded.request(
            "POST", "/v1/responses/compact", {"model": _MODEL, "input": _input()}
        ), "/responses/compact"
    upstream_id: Final = f"resp_{uuid.uuid4().hex}"
    managed_id: Final = _create(guarded, provider, upstream_id)
    provider.expect(_deleted(upstream_id) if operation == "delete" else _response(upstream_id))
    if operation == "get":
        return guarded.request("GET", f"/v1/responses/{managed_id}"), f"/responses/{upstream_id}"
    if operation == "delete":
        return guarded.request("DELETE", f"/v1/responses/{managed_id}"), f"/responses/{upstream_id}"
    return guarded.request("POST", f"/v1/responses/{managed_id}/cancel"), f"/responses/{upstream_id}/cancel"


@pytest.mark.parametrize("operation", ["get", "compact", "delete", "cancel"])
def test_responses_lifecycle_answers_200_with_a_during_call_guardrail_on(
    guarded: Gateway, provider: SharedProvider, operation: str
) -> None:
    response, upstream_target = _lifecycle(guarded, provider, operation)
    assert response.status_code == 200, response.text
    (upstream,) = provider.received()
    assert upstream.target.endswith(upstream_target), upstream.target
