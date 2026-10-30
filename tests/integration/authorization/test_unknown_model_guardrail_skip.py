"""LIT-8011: a model name the proxy cannot route gets the same ``400 Invalid model name``
it gets with no guardrail, and a ``default_on`` pre_call guardrail never sends it to the
vendor. A routable model is still scanned before the provider call, and a guardrail from the
operator's own module still runs, along with every guardrail listed before it.
"""

import json
import uuid
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_MODEL_TEXT: Final = "routed reply"
_VERDICTS: Final = {
    "allow": Reply(body=json.dumps({"action": "allow", "category": "benign", "response_detected": {}}).encode()),
    "block": Reply(
        body=json.dumps(
            {"action": "block", "category": "malicious", "prompt_detected": {"dlp": True}, "response_detected": {}}
        ).encode()
    ),
    "unreachable": Reply(status=503, body=b'{"error": "scanner down"}'),
}


def _scanner(verdict: str) -> Callable[[Request], Reply]:
    def reply(request: Request) -> Reply:
        assert request.target == "/v1/scan/sync/request"
        return _VERDICTS[verdict]

    return reply


def _provider(request: Request) -> Reply:
    if request.method == "GET" and request.target.endswith("/models"):
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    assert request.target == "/v1/chat/completions", request.target
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl_synthetic",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4.1-mini",
                "choices": [
                    {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": _MODEL_TEXT}}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
            }
        ).encode()
    )


def _operator_verdict(action: str) -> Callable[[Request], Reply]:
    def reply(request: Request) -> Reply:
        return Reply(body=json.dumps({"action": action}).encode())

    return reply


def _airs(policy_url: str) -> dict[str, JsonValue]:
    return {
        "guardrail_name": "airs-" + uuid.uuid4().hex[:8],
        "litellm_params": {
            "guardrail": "panw_prisma_airs",
            "mode": "pre_call",
            "default_on": True,
            "api_base": policy_url,
            "api_key": "synthetic-panw-key",
            "profile_name": "synthetic-profile",
        },
    }


def _operator_guardrail() -> dict[str, JsonValue]:
    return {
        "guardrail_name": "operator-" + uuid.uuid4().hex[:8],
        "litellm_params": {
            "guardrail": "integration.authorization._operator_guardrail.OperatorGuardrail",
            "mode": "pre_call",
            "default_on": True,
        },
    }


def _config(tmp_path: Path, *guardrails: Mapping[str, JsonValue]) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = list(guardrails)
    path: Final = tmp_path / "pre_call_guardrails.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _requests(model: str) -> tuple[tuple[str, dict[str, JsonValue]], ...]:
    messages: Final = [{"role": "user", "content": "my card is 4111 1111 1111 1111"}]
    return (
        ("/v1/chat/completions", {"model": model, "messages": messages}),
        ("/v1/chat/completions", {"model": model, "messages": messages, "stream": True}),
        ("/v1/responses", {"model": model, "input": "my card is 4111 1111 1111 1111"}),
        ("/v1/messages", {"model": model, "max_tokens": 16, "messages": messages}),
        ("/v1/embeddings", {"model": model, "input": "my card is 4111 1111 1111 1111"}),
        ("/v1/completions", {"model": model, "prompt": "my card is 4111 1111 1111 1111"}),
    )


@pytest.mark.parametrize("verdict", sorted(_VERDICTS))
def test_unknown_model_gets_routing_400_without_calling_pre_call_guardrail(
    gateway: Gateway, tmp_path: Path, verdict: str
) -> None:
    model: Final = "no-such-model-" + uuid.uuid4().hex[:8]
    with wire_server(_scanner(verdict)) as policy:
        with owned_proxy(gateway, tmp_path, {}, config=_config(tmp_path, _airs(policy.url))) as candidate:
            for target, body in _requests(model):
                control = gateway.request("POST", target, body)
                response = candidate.request("POST", target, body)

                assert control.status_code == 400, control.text
                assert "Invalid model name" in control.text
                assert (response.status_code, response.json()) == (control.status_code, control.json()), target
            assert policy.drain() == ()


def test_routable_model_is_still_scanned_before_the_provider_call(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_scanner("allow")) as policy, wire_server(_provider) as upstream:
        with owned_proxy(gateway, tmp_path, {}, config=_config(tmp_path, _airs(policy.url))) as candidate:
            with candidate.scenario() as scenario:
                model: Final = scenario.model(
                    model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
                )
                response: Final = candidate.request("POST", "/v1/chat/completions", dict(_requests(model)[0][1]))

                assert response.status_code == 200, response.text
                assert response.json()["choices"][0]["message"]["content"] == _MODEL_TEXT
                scans: Final = policy.drain()
                assert len(scans) == 1
                assert "4111" in json.loads(scans[0].body)["contents"][0]["prompt"]


def test_routable_model_block_verdict_still_blocks(gateway: Gateway, tmp_path: Path) -> None:
    with wire_server(_scanner("block")) as policy, wire_server(_provider) as upstream:
        with owned_proxy(gateway, tmp_path, {}, config=_config(tmp_path, _airs(policy.url))) as candidate:
            with candidate.scenario() as scenario:
                model: Final = scenario.model(
                    model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
                )
                response: Final = candidate.request("POST", "/v1/chat/completions", dict(_requests(model)[0][1]))

                assert response.status_code == 400, response.text
                assert "Invalid model name" not in response.text
                assert len(policy.drain()) == 1
                assert [request.target for request in upstream.drain() if request.method == "POST"] == []


def test_unknown_model_burst_while_the_scanner_is_down_never_reaches_it(gateway: Gateway, tmp_path: Path) -> None:
    model: Final = "no-such-model-" + uuid.uuid4().hex[:8]
    unknown: Final = _requests(model)
    controls: Final = tuple(gateway.request("POST", target, body) for target, body in unknown)
    with wire_server(_scanner("unreachable")) as policy, wire_server(_provider) as upstream:
        with owned_proxy(gateway, tmp_path, {}, config=_config(tmp_path, _airs(policy.url)), workers=2) as candidate:
            with candidate.scenario() as scenario:
                routable: Final = scenario.model(
                    model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
                )
                known: Final = (
                    "/v1/chat/completions",
                    {"model": routable, "messages": [{"role": "user", "content": "routable burst prompt"}]},
                )
                burst: Final = unknown * 6 + (known,) * 6
                with ThreadPoolExecutor(max_workers=16) as pool:
                    responses: Final = tuple(pool.map(lambda item: candidate.request("POST", *item), burst))

                for index, response in enumerate(responses[: len(unknown) * 6]):
                    control = controls[index % len(unknown)]
                    assert (response.status_code, response.json()) == (control.status_code, control.json()), index
                known_statuses: Final = {response.status_code for response in responses[len(unknown) * 6 :]}
                assert 200 not in known_statuses, known_statuses
                scanned: Final = [json.loads(scan.body)["contents"][0]["prompt"] for scan in policy.drain()]
                assert scanned and all("routable burst prompt" in prompt for prompt in scanned), scanned
                assert [request.target for request in upstream.drain() if request.method == "POST"] == []


def test_operator_guardrail_and_the_guardrail_before_it_still_run_for_an_unknown_model(
    gateway: Gateway, tmp_path: Path
) -> None:
    model: Final = "no-such-model-" + uuid.uuid4().hex[:8]
    with wire_server(_scanner("allow")) as policy, wire_server(_operator_verdict("block")) as operator:
        config: Final = _config(tmp_path, _airs(policy.url), _operator_guardrail())
        with owned_proxy(gateway, tmp_path, {"OPERATOR_GUARDRAIL_URL": operator.url}, config=config) as candidate:
            response: Final = candidate.request("POST", "/v1/chat/completions", dict(_requests(model)[0][1]))

            assert response.status_code == 403, response.text
            assert "blocked by the operator guardrail" in response.text
            assert len(policy.drain()) == 1
            assert [json.loads(request.body) for request in operator.drain()] == [{"model": model}]


def test_built_in_guardrail_after_an_allowing_operator_guardrail_is_skipped_for_an_unknown_model(
    gateway: Gateway, tmp_path: Path
) -> None:
    model: Final = "no-such-model-" + uuid.uuid4().hex[:8]
    body: Final = dict(_requests(model)[0][1])
    control: Final = gateway.request("POST", "/v1/chat/completions", body)
    with wire_server(_scanner("block")) as policy, wire_server(_operator_verdict("allow")) as operator:
        config: Final = _config(tmp_path, _operator_guardrail(), _airs(policy.url))
        with owned_proxy(gateway, tmp_path, {"OPERATOR_GUARDRAIL_URL": operator.url}, config=config) as candidate:
            response: Final = candidate.request("POST", "/v1/chat/completions", body)

            assert (response.status_code, response.json()) == (control.status_code, control.json())
            assert [json.loads(request.body) for request in operator.drain()] == [{"model": model}]
            assert policy.drain() == ()
