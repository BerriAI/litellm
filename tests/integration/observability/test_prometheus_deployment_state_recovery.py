from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment
from integration._support.process import owned_proxy
from integration._support.prometheus_series import Sample, scrape
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

RUN: Final = uuid.uuid4().hex[:8]
STATE: Final = "litellm_deployment_state"
GUARDRAIL: Final = f"deployment-state-filter-{RUN}"
APPLIED_GUARDRAILS: Final = "x-litellm-applied-guardrails"


@dataclass(frozen=True, slots=True)
class Cell:
    name: str
    stream: bool
    guardrail: str | None

    @property
    def group(self) -> str:
        return f"state-group-{self.name}-{RUN}"

    @property
    def model(self) -> str:
        return f"state-model-{self.name}-{RUN}"

    @property
    def model_id(self) -> str:
        return f"state-deployment-{self.name}-{RUN}"


CELLS: Final = (
    Cell("chat", stream=False, guardrail=None),
    Cell("chat-stream", stream=True, guardrail=None),
    Cell("guarded-chat", stream=False, guardrail=GUARDRAIL),
    Cell("guarded-chat-stream", stream=True, guardrail=GUARDRAIL),
)


class Provider:
    def __init__(self) -> None:
        self._lock: Final = threading.Lock()
        self._rejecting: frozenset[str] = frozenset()

    def reject(self, model: str) -> None:
        with self._lock:
            self._rejecting = self._rejecting | {model}

    def recover(self, model: str) -> None:
        with self._lock:
            self._rejecting = self._rejecting - {model}

    def __call__(self, request: Request) -> Reply:
        sent: Final = json.loads(request.body)
        with self._lock:
            rejecting: Final = sent["model"] in self._rejecting
        if rejecting:
            error: Final = {"message": "Incorrect API key provided", "type": "invalid_request_error"}
            return Reply(status=401, body=json.dumps({"error": {**error, "code": "invalid_api_key"}}).encode())
        if sent.get("stream"):
            return Reply(content_type="text/event-stream", chunks=_chat_frames(sent["model"]))
        return Reply(body=json.dumps(_completion(sent["model"])).encode())


def _completion(model: str) -> dict[str, JsonValue]:
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": 1,
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4},
    }


def _chat_frames(model: str) -> tuple[bytes, ...]:
    chunk: Final = {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
    }
    frames: Final = (
        {**chunk, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "ok"}, "finish_reason": None}]},
        {**chunk, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
    )
    return (*(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames), b"data: [DONE]\n\n")


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    upstream: Wire
    provider: Provider


def _config(directory: Path, upstream: Wire) -> Path:
    configuration: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    configuration["model_list"] = [
        *configuration["model_list"],
        *(
            {
                "model_name": cell.group,
                "litellm_params": {
                    "model": f"openai/{cell.model}",
                    "api_key": "sk-deployment-own-key",
                    "api_base": f"{upstream.url}/v1",
                },
                "model_info": {"id": cell.model_id},
            }
            for cell in CELLS
        ),
    ]
    configuration["guardrails"] = [
        {
            "guardrail_name": GUARDRAIL,
            "litellm_params": {
                "guardrail": "litellm_content_filter",
                "mode": "pre_call",
                "default_on": False,
                "blocked_words": [{"keyword": f"blocked-{RUN}", "action": "BLOCK"}],
            },
        }
    ]
    configuration["litellm_settings"]["callbacks"] = ["prometheus"]
    configuration.setdefault("router_settings", {}).update(
        {"num_retries": 0, "disable_cooldowns": False, "cooldown_time": 1}
    )
    path: Final = directory / "prometheus-deployment-state-recovery.yaml"
    path.write_text(yaml.safe_dump(configuration))
    return path


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Rig]:
    directory: Final = tmp_path_factory.mktemp("prometheus-deployment-state-recovery")
    provider: Final = Provider()
    with (
        wire_server(provider) as upstream,
        gateway_from_environment() as shared,
        owned_proxy(shared, directory, {}, config=_config(directory, upstream)) as owned,
    ):
        yield Rig(owned, upstream, provider)


def _send(rig: Rig, cell: Cell) -> httpx.Response:
    body: Final[dict[str, JsonValue]] = {
        "model": cell.group,
        "messages": [{"role": "user", "content": f"deployment state {uuid.uuid4().hex}"}],
        "stream": cell.stream,
        **({"guardrails": [cell.guardrail]} if cell.guardrail else {}),
    }
    return rig.gateway.request("POST", "/v1/chat/completions", body)


def _states(samples: Sequence[Sample], model_id: str) -> list[float]:
    return sorted(sample.value for sample in samples if sample.name == STATE and sample.labels["model_id"] == model_id)


def _states_become(rig: Rig, cell: Cell, expected: Sequence[float]) -> Mapping[str, list[float]]:
    return eventually(
        lambda: {"states": _states(scrape(rig.gateway), cell.model_id)},
        lambda observed: observed["states"] == list(expected),
        seconds=30,
    )


@pytest.mark.parametrize("cell", CELLS, ids=[cell.name for cell in CELLS])
def test_the_deployment_state_returns_to_healthy_once_the_provider_recovers_from_a_cooldown(
    rig: Rig, cell: Cell
) -> None:
    healthy: Final = _send(rig, cell)
    assert healthy.status_code == 200, healthy.text
    assert not cell.stream or "data: [DONE]" in healthy.text, healthy.text
    if cell.guardrail:
        assert healthy.headers.get(APPLIED_GUARDRAILS) == cell.guardrail, dict(healthy.headers)
    _states_become(rig, cell, [0.0])

    rig.provider.reject(cell.model)
    rejected: Final = _send(rig, cell)
    assert rejected.status_code == 401, rejected.text
    _states_become(rig, cell, [2.0])

    rig.provider.recover(cell.model)
    recovered: Final = eventually(lambda: _send(rig, cell), lambda response: response.status_code == 200, seconds=30)
    if cell.guardrail:
        assert recovered.headers.get(APPLIED_GUARDRAILS) == cell.guardrail, dict(recovered.headers)
    _states_become(rig, cell, [0.0])
