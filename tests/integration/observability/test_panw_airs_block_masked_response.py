"""LIT-9336: ``include_masked_response_on_block`` decides whether the 400 a PANW
post_call block raises also carries AIRS ``response_masked_data`` (masked text
plus ``pattern_detections``) back to the caller.

The double returns allow on prompt scans and a customer-shaped block body on
response scans, so every denial here is a post_call block.
"""

import json
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, object_value
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, wire_server
from pydantic import JsonValue

_MODEL_TEXT: Final = "sensitive generation"


def _masked_data() -> dict[str, JsonValue]:
    return {
        "data": "Sample (fictional) values:\n\n- Email ID: `XXXXXXXXXXXXXXXXXXXXX`\n- SSN: `XXXXXXXXXXX`",
        "pattern_detections": [
            {"locations": [[72, 74], [76, 82]], "pattern": "License Plate Number - USA"},
            {"locations": [[72, 74], [79, 82]], "pattern": "Credit Card CVV"},
            {"locations": [[72, 82]], "pattern": "Driver License - US"},
            {"locations": [[41, 61]], "pattern": "Internet - addr_email"},
        ],
    }


def _block_scan(with_masked_data: bool = True) -> dict[str, JsonValue]:
    scan_id: Final = uuid.uuid4().hex
    session_id: Final = uuid.uuid4().hex
    return {
        "action": "block",
        "category": "malicious",
        "error": False,
        "errors": [],
        "profile_id": "a6ecc0c6-9aee-41bf-b967-22c690a8b989",
        "profile_name": "DLP_PII_NexusOS",
        "prompt_detected": {},
        "report_id": "R" + scan_id,
        "response_detected": {"dlp": True, "toxic_content": False, "ungrounded": False},
        "scan_id": scan_id,
        "session_id": session_id,
        "source": "AI-Runtime-API",
        "timeout": False,
        "tool_detected": {},
        "tr_id": session_id,
        "transaction_id": "pan_" + uuid.uuid4().hex,
        **({"response_masked_data": _masked_data()} if with_masked_data else {}),
    }


def _scanner(request: Request) -> Reply:
    assert request.target == "/v1/scan/sync/request"
    contents: Final = json.loads(request.body)["contents"][0]
    if "response" in contents:
        return Reply(body=json.dumps(_block_scan()).encode())
    return Reply(
        body=json.dumps({"action": "allow", "category": "benign", "response_detected": {}}).encode()
    )


def _scanner_without_masked_data(request: Request) -> Reply:
    assert request.target == "/v1/scan/sync/request"
    contents: Final = json.loads(request.body)["contents"][0]
    if "response" in contents:
        return Reply(body=json.dumps(_block_scan(with_masked_data=False)).encode())
    return Reply(
        body=json.dumps({"action": "allow", "category": "benign", "response_detected": {}}).encode()
    )


def _provider(request: Request) -> Reply:
    if request.method == "GET" and request.target.endswith("/models"):
        return Reply(body=json.dumps({"object": "list", "data": []}).encode())
    if request.target == "/v1/chat/completions":
        return Reply(
            body=json.dumps(
                {
                    "id": "chatcmpl_synthetic",
                    "object": "chat.completion",
                    "created": 1700000000,
                    "model": "gpt-4.1-mini",
                    "choices": [
                        {
                            "index": 0,
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": _MODEL_TEXT},
                        }
                    ],
                    "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
                }
            ).encode()
        )
    if request.target == "/v1/responses":
        return Reply(
            body=json.dumps(
                {
                    "id": "resp_" + uuid.uuid4().hex,
                    "object": "response",
                    "created_at": 1700000000,
                    "status": "completed",
                    "model": "gpt-4.1-mini",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_synthetic",
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": _MODEL_TEXT, "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
                }
            ).encode()
        )
    if request.target == "/v1/messages":
        return Reply(
            body=json.dumps(
                {
                    "id": "msg_" + uuid.uuid4().hex,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [{"type": "text", "text": _MODEL_TEXT}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 4},
                }
            ).encode()
        )
    raise AssertionError(f"unexpected provider target {request.target}")


def _config(tmp_path: Path, identity: str, policy_url: str, **flags: bool) -> Path:
    config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
    config["guardrails"] = [
        {
            "guardrail_name": identity,
            "litellm_params": {
                "guardrail": "panw_prisma_airs",
                "mode": "post_call",
                "default_on": True,
                "api_base": policy_url,
                "api_key": "synthetic-panw-key",
                "profile_name": "synthetic-profile",
                **flags,
            },
        }
    ]
    path: Final = tmp_path / "panw_block.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def _block_detail(node: JsonValue) -> Mapping[str, JsonValue] | None:
    if isinstance(node, dict):
        if node.get("code") == "panw_prisma_airs_response_blocked":
            return node
        for value in node.values():
            found: Final = _block_detail(value)
            if found is not None:
                return found
    if isinstance(node, list):
        for value in node:
            found: Final = _block_detail(value)
            if found is not None:
                return found
    return None


def _response_scans(requests: tuple[Request, ...]) -> list[dict[str, JsonValue]]:
    return [
        contents
        for request in requests
        for contents in json.loads(request.body)["contents"]
        if "response" in contents
    ]


def _request_body(model: str, target: str) -> dict[str, JsonValue]:
    if target == "/v1/responses":
        return {"model": model, "input": "say hi"}
    if target == "/v1/messages":
        return {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": "say hi"}]}
    return {"model": model, "messages": [{"role": "user", "content": "say hi"}]}


@pytest.mark.parametrize("target", ["/v1/chat/completions", "/v1/responses", "/v1/messages"])
def test_block_includes_response_masked_data_when_opted_in(gateway: Gateway, tmp_path: Path, target: str) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    with wire_server(_scanner) as policy, wire_server(_provider) as upstream:
        config: Final = _config(tmp_path, identity, policy.url, include_masked_response_on_block=True)
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            response: Final = candidate.request("POST", target, _request_body(model, target))

            assert response.status_code == 400, response.text
            detail: Final = _block_detail(object_value(response.json()))
            assert detail is not None, f"blocked detail missing from {response.text}"
            assert detail["action"] == "block"
            assert detail["category"] == "malicious"
            assert detail["response_masked_data"] == _masked_data()
            assert _response_scans(policy.drain()) != [], "the AIRS double never saw a response scan"


def test_block_withholds_response_masked_data_by_default(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    with wire_server(_scanner) as policy, wire_server(_provider) as upstream:
        config: Final = _config(tmp_path, identity, policy.url)
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            response: Final = candidate.request(
                "POST", "/v1/chat/completions", _request_body(model, "/v1/chat/completions")
            )

            assert response.status_code == 400, response.text
            detail: Final = _block_detail(object_value(response.json()))
            assert detail is not None, f"blocked detail missing from {response.text}"
            assert "response_masked_data" not in detail
            assert detail["action"] == "block"
            assert _response_scans(policy.drain()) != []


def test_block_without_masked_data_invents_none(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    with wire_server(_scanner_without_masked_data) as policy, wire_server(_provider) as upstream:
        config: Final = _config(tmp_path, identity, policy.url, include_masked_response_on_block=True)
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            response: Final = candidate.request(
                "POST", "/v1/chat/completions", _request_body(model, "/v1/chat/completions")
            )

            assert response.status_code == 400, response.text
            detail: Final = _block_detail(object_value(response.json()))
            assert detail is not None, f"blocked detail missing from {response.text}"
            assert "response_masked_data" not in detail
            assert _MODEL_TEXT not in response.text


def test_mask_response_content_still_masks_when_flag_on(gateway: Gateway, tmp_path: Path) -> None:
    identity: Final = "guardrail" + uuid.uuid4().hex
    with wire_server(_scanner) as policy, wire_server(_provider) as upstream:
        config: Final = _config(
            tmp_path,
            identity,
            policy.url,
            include_masked_response_on_block=True,
            mask_response_content=True,
        )
        with owned_proxy(gateway, tmp_path, {}, config=config) as candidate, candidate.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            response: Final = candidate.request(
                "POST", "/v1/chat/completions", _request_body(model, "/v1/chat/completions")
            )

            assert response.status_code == 200, response.text
            content: Final = response.json()["choices"][0]["message"]["content"]
            assert content == _masked_data()["data"]


def test_db_created_guardrail_string_false_withholds_response_masked_data_on_block(gateway: Gateway) -> None:
    """The dashboard persists some litellm_params as strings; "false" must not turn the flag on."""
    name: Final = f"panw-string-false-{uuid.uuid4().hex[:8]}"
    with wire_server(_scanner) as policy, wire_server(_provider) as upstream:
        gateway.post(
            "/guardrails",
            {
                "guardrail": {
                    "guardrail_name": name,
                    "litellm_params": {
                        "guardrail": "panw_prisma_airs",
                        "mode": "post_call",
                        "api_base": policy.url,
                        "api_key": "synthetic-panw-key",
                        "profile_name": "synthetic-profile",
                        "include_masked_response_on_block": "false",
                    },
                }
            },
        )
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {**_request_body(model, "/v1/chat/completions"), "guardrails": [name]},
            )

            assert response.status_code == 400, response.text
            detail: Final = _block_detail(object_value(response.json()))
            assert detail is not None, f"blocked detail missing from {response.text}"
            assert "response_masked_data" not in detail
            assert _response_scans(policy.drain()) != [], "the AIRS double never saw a response scan"


def test_db_created_guardrail_includes_response_masked_data_on_block(gateway: Gateway, tmp_path: Path) -> None:
    name: Final = f"panw-block-details-{uuid.uuid4().hex[:8]}"
    with wire_server(_scanner) as policy, wire_server(_provider) as upstream:
        gateway.post(
            "/guardrails",
            {
                "guardrail": {
                    "guardrail_name": name,
                    "litellm_params": {
                        "guardrail": "panw_prisma_airs",
                        "mode": "post_call",
                        "api_base": policy.url,
                        "api_key": "synthetic-panw-key",
                        "profile_name": "synthetic-profile",
                        "include_masked_response_on_block": True,
                    },
                }
            },
        )
        with gateway.scenario() as scenario:
            model: Final = scenario.model(
                model="openai/gpt-4.1-mini", api_base=upstream.url + "/v1", api_key="synthetic-key"
            )
            response: Final = gateway.request(
                "POST",
                "/v1/chat/completions",
                {**_request_body(model, "/v1/chat/completions"), "guardrails": [name]},
            )

            assert response.status_code == 400, response.text
            detail: Final = _block_detail(object_value(response.json()))
            assert detail is not None, f"blocked detail missing from {response.text}"
            assert detail["action"] == "block"
            assert detail["category"] == "malicious"
            assert detail["response_masked_data"] == _masked_data()
            assert _response_scans(policy.drain()) != [], "the AIRS double never saw a response scan"
