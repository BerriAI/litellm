import json
import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server

_ATTACK_MARKER: Final = "synthetic-attack-marker"

_AZURE_TARGET_PREFIX: Final = "/contentsafety/text:shieldPrompt?api-version="


def _azure_shield(request: Request) -> Reply:
    assert request.method == "POST"
    assert request.target.startswith(_AZURE_TARGET_PREFIX), request.target
    user_prompt: Final = object_value(json.loads(request.body))["userPrompt"]
    assert isinstance(user_prompt, str)
    return Reply(
        body=json.dumps(
            {
                "userPromptAnalysis": {"attackDetected": _ATTACK_MARKER in user_prompt},
                "documentsAnalysis": [],
            }
        ).encode()
    )


def _provider(request: Request) -> Reply:
    assert request.method == "POST"
    if request.target == "/v1/messages":
        return Reply(
            body=json.dumps(
                {
                    "id": "msg_" + uuid.uuid4().hex,
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-5-20250929",
                    "content": [{"type": "text", "text": "permitted response"}],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 11, "output_tokens": 4},
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
                            "id": "msg_" + uuid.uuid4().hex,
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "permitted response", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 11, "output_tokens": 4, "total_tokens": 15},
                }
            ).encode()
        )
    assert request.target == "/v1/chat/completions", request.target
    return Reply(
        body=json.dumps(
            {
                "id": "chatcmpl-" + uuid.uuid4().hex,
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4.1-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "permitted response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 11, "completion_tokens": 4, "total_tokens": 15},
            }
        ).encode()
    )


@pytest.fixture(scope="module")
def azure_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Wire, Wire]]:
    directory: Final = tmp_path_factory.mktemp("azure-shield")
    with ExitStack() as stack:
        gateway: Final = stack.enter_context(gateway_from_environment())
        azure: Final = stack.enter_context(wire_server(_azure_shield))
        provider: Final = stack.enter_context(wire_server(_provider))
        config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        config["guardrails"] = [
            {
                "guardrail_name": "azure-shield-" + uuid.uuid4().hex,
                "litellm_params": {
                    "guardrail": "azure/prompt_shield",
                    "mode": "pre_call",
                    "default_on": True,
                    "api_base": azure.url,
                    "api_key": "synthetic-azure-key",
                    "cost_tier": "paid",
                    "price_per_1000_text_records": 0.38,
                },
            }
        ]
        path: Final = directory / "azure-shield.yaml"
        path.write_text(yaml.safe_dump(config))
        candidate: Final = stack.enter_context(owned_proxy(gateway, directory, {}, config=path))
        yield candidate, azure, provider


@pytest.fixture(autouse=True)
def _clear_wires(azure_rig: tuple[Gateway, Wire, Wire]) -> None:
    azure_rig[1].drain()
    azure_rig[2].drain()


def _scanned_prompts(azure: Wire) -> list[str]:
    return [
        object_value(json.loads(scan.body))["userPrompt"]
        for scan in azure.drain()
        if scan.target.startswith(_AZURE_TARGET_PREFIX)
    ]


def _guardrail_entry(model: str) -> dict:
    rows: Final = eventually(
        lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    saved: Final = object_value(rows[0]["metadata"])
    entries: Final = saved["guardrail_information"]
    assert isinstance(entries, list) and len(entries) == 1, saved
    return object_value(entries[0])


@pytest.mark.parametrize(
    ("path", "body_shape", "model_provider"),
    [
        pytest.param("/v1/chat/completions", "chat", "openai", id="chat-completions-messages"),
        pytest.param("/v1/messages", "chat", "anthropic", id="anthropic-messages"),
        pytest.param("/v1/responses", "responses-string", "openai", id="responses-string-input"),
        pytest.param("/v1/responses", "responses-list", "openai", id="responses-list-input"),
        pytest.param(
            "/v1/responses", "responses-string-with-empty-messages", "openai", id="responses-empty-messages-stub"
        ),
    ],
)
def test_azure_prompt_shield_scans_the_user_prompt_on_every_endpoint(
    azure_rig: tuple[Gateway, Wire, Wire], path: str, body_shape: str, model_provider: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"synthetic prompt {body_shape} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = scenario.model(
            model=("anthropic/claude-sonnet-4-5-20250929" if model_provider == "anthropic" else "openai/gpt-4.1-mini"),
            api_base=provider.url if model_provider == "anthropic" else provider.url + "/v1",
            api_key="synthetic-provider-key",
        )
        body: Final = {
            "responses-string": {"input": prompt},
            "responses-list": {"input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}]},
            "responses-string-with-empty-messages": {"messages": [], "input": prompt},
        }.get(
            body_shape,
            {"messages": [{"role": "user", "content": prompt}], "max_tokens": 16},
        )
        response: Final = candidate.request("POST", path, {"model": model, **body})
        assert response.status_code == 200, response.text
        assert "permitted response" in response.text
        assert _scanned_prompts(azure) == [prompt]
        assert len(provider.drain()) == 1
        entry: Final = _guardrail_entry(model)
        assert entry["guardrail_status"] == "success", entry
        assert entry["guardrail_usage"] == {
            "requests": 1,
            "input_characters": len(prompt),
            "text_records": 1,
        }, entry
        assert entry["guardrail_cost"] == pytest.approx(0.38 / 1000), entry


def test_azure_prompt_shield_blocks_attack_in_responses_input(
    azure_rig: tuple[Gateway, Wire, Wire],
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"synthetic prompt {_ATTACK_MARKER} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = scenario.model(
            model="openai/gpt-4.1-mini",
            api_base=provider.url + "/v1",
            api_key="synthetic-provider-key",
        )
        response: Final = candidate.request("POST", "/v1/responses", {"model": model, "input": prompt})
        assert response.status_code == 400, response.text
        assert "Violated Azure Prompt Shield guardrail policy" in response.text
        assert _scanned_prompts(azure) == [prompt]
        assert provider.drain() == ()
