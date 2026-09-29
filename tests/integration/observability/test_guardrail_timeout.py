import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import httpx
import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server

GUARDRAIL_TIMEOUT: Final = 2
VENDOR_STALL: Final = 25
CALLER_TIMEOUT: Final = 60
PROMPT: Final = "synthetic moderation prompt"
PROVIDER_TEXT: Final = "provider answer"


def _moderation_reply() -> Reply:
    return Reply(
        body=json.dumps(
            {
                "id": "modr-" + uuid.uuid4().hex,
                "model": "omni-moderation-latest",
                "results": [
                    {"flagged": False, "categories": {}, "category_scores": {}, "category_applied_input_types": {}}
                ],
            }
        ).encode()
    )


def _provider(request: Request) -> Reply:
    assert request.target == "/v1/chat/completions"
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1,
                "model": "gpt-4o-mini",
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": PROVIDER_TEXT}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 7, "completion_tokens": 2, "total_tokens": 9},
            }
        ).encode()
    )


@dataclass(frozen=True, slots=True)
class Rig:
    gateway: Gateway
    model: str
    provider: Wire
    stalled: Wire
    failing: Wire
    healthy: Wire

    def chat(self, guardrail: str) -> tuple[httpx.Response, float]:
        started: Final = time.monotonic()
        response: Final = self.gateway.client.post(
            "/v1/chat/completions",
            json={"model": self.model, "guardrails": [guardrail], "messages": [{"role": "user", "content": PROMPT}]},
            headers={"Authorization": f"Bearer {self.gateway.key}"},
            timeout=CALLER_TIMEOUT,
        )
        return response, time.monotonic() - started


def _guardrail(name: str, vendor: Wire, **params: object) -> dict[str, object]:
    return {
        "guardrail_name": name,
        "litellm_params": {
            "guardrail": "openai_moderation",
            "mode": "pre_call",
            "api_base": f"{vendor.url}/v1",
            "api_key": "synthetic-moderation-key",
            **params,
        },
    }


@pytest.fixture
def rig(gateway: Gateway, tmp_path: Path) -> Iterator[Rig]:
    release: Final = threading.Event()

    def stall(request: Request) -> Reply:
        release.wait(timeout=VENDOR_STALL)
        return _moderation_reply()

    def fail(request: Request) -> Reply:
        assert request.target == "/v1/moderations"
        return Reply(status=500, body=b'{"error": "synthetic moderation outage"}')

    def answer(request: Request) -> Reply:
        assert request.target == "/v1/moderations"
        assert json.loads(request.body)["input"] == PROMPT
        return _moderation_reply()

    with ExitStack() as stack:
        provider: Final = stack.enter_context(wire_server(_provider))
        stalled: Final = stack.enter_context(wire_server(stall))
        failing: Final = stack.enter_context(wire_server(fail))
        healthy: Final = stack.enter_context(wire_server(answer))
        stack.callback(release.set)
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            _guardrail("stalled-fail-closed", stalled, timeout=GUARDRAIL_TIMEOUT),
            _guardrail("stalled-fail-open", stalled, timeout=GUARDRAIL_TIMEOUT, unreachable_fallback="fail_open"),
            _guardrail("failing-fail-open", failing, timeout=GUARDRAIL_TIMEOUT, unreachable_fallback="fail_open"),
            _guardrail("healthy-bounded", healthy, timeout=GUARDRAIL_TIMEOUT),
            {
                "guardrail_name": "singulr-open-on-error",
                "litellm_params": {
                    "guardrail": "singulr",
                    "mode": "pre_call",
                    "api_base": stalled.url,
                    "api_key": "synthetic-singulr-key",
                    "block_on_error": False,
                    "timeout": GUARDRAIL_TIMEOUT,
                },
            },
            {
                "guardrail_name": "custom-code-spinning",
                "litellm_params": {
                    "guardrail": "custom_code",
                    "mode": "pre_call",
                    "timeout": GUARDRAIL_TIMEOUT,
                    "custom_code": "def apply_guardrail(inputs, request_data, input_type):\n    while True:\n        pass\n",
                },
            },
        ]
        path: Final = tmp_path / "guardrail-timeout.yaml"
        path.write_text(yaml.safe_dump(config))
        candidate: Final = stack.enter_context(owned_proxy(gateway, tmp_path, {}, config=path))
        scenario: Final = stack.enter_context(candidate.scenario())
        model: Final = scenario.model(api_base=f"{provider.url}/v1", api_key="synthetic-provider-key")
        yield Rig(candidate, model, provider, stalled, failing, healthy)


def test_stalled_guardrail_fails_closed_with_408_at_its_timeout(rig: Rig) -> None:
    response, elapsed = rig.chat("stalled-fail-closed")

    assert len(rig.stalled.drain()) == 1
    assert response.status_code == 408, (response.status_code, round(elapsed, 2), response.text)
    assert "Guardrail 'stalled-fail-closed' did not finish within 2.0s" in response.text
    assert elapsed < GUARDRAIL_TIMEOUT + 5
    assert rig.provider.drain() == ()


def test_stalled_guardrail_fails_open_and_the_model_answers(rig: Rig) -> None:
    response, elapsed = rig.chat("stalled-fail-open")

    assert len(rig.stalled.drain()) == 1
    assert elapsed < GUARDRAIL_TIMEOUT + 5, (response.status_code, round(elapsed, 2), response.text)
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == PROVIDER_TEXT
    assert len(rig.provider.drain()) == 1


def test_fast_guardrail_error_is_not_turned_into_fail_open(rig: Rig) -> None:
    response, elapsed = rig.chat("failing-fail-open")

    assert len(rig.failing.drain()) == 1
    assert response.status_code == 500, response.text
    assert "synthetic moderation outage" in response.text
    assert elapsed < GUARDRAIL_TIMEOUT
    assert rig.provider.drain() == ()


def test_healthy_guardrail_under_its_timeout_lets_the_request_through(rig: Rig) -> None:
    response, _ = rig.chat("healthy-bounded")

    assert len(rig.healthy.drain()) == 1
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == PROVIDER_TEXT
    assert len(rig.provider.drain()) == 1


def test_guardrail_applying_timeout_to_its_own_http_call_still_fails_closed(rig: Rig) -> None:
    response, elapsed = rig.chat("singulr-open-on-error")

    assert [request.target for request in rig.stalled.drain()] == ["/api/v1/ai-gateway/litellm-v2"]
    assert response.status_code == 408, (response.status_code, round(elapsed, 2), response.text)
    assert elapsed < GUARDRAIL_TIMEOUT + 5
    assert rig.provider.drain() == ()


def test_custom_code_past_its_timeout_keeps_its_own_execution_timeout_error(rig: Rig) -> None:
    response, elapsed = rig.chat("custom-code-spinning")

    assert response.status_code == 500, (response.status_code, round(elapsed, 2), response.text)
    assert "Custom code guardrail 'custom-code-spinning' exceeded its 2s execution timeout" in response.text
    assert elapsed < GUARDRAIL_TIMEOUT + 5
    assert rig.provider.drain() == ()
