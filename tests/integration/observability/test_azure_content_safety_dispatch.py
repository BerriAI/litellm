import json
import uuid
from collections.abc import Iterator
from contextlib import ExitStack
from pathlib import Path
from typing import Final

import pytest
import yaml
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, object_value
from integration._support.database import read_rows
from integration._support.process import owned_proxy
from integration._support.wire import Reply, Request, Wire, wire_server
from pydantic import JsonValue

_ATTACK_MARKER: Final = "synthetic-attack-marker"
_MODERATION_MARKER: Final = "synthetic-moderation-marker"
_SHIELD_TARGET_PREFIX: Final = "/contentsafety/text:shieldPrompt?api-version="
_ANALYZE_TARGET_PREFIX: Final = "/contentsafety/text:analyze?api-version="
_HOOKS_SOURCE: Final = """from __future__ import annotations

from typing import Final, cast

from fastapi import HTTPException

from litellm.caching.caching import DualCache
from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.guardrails.guardrail_hooks.azure.prompt_shield import (
    AzureContentSafetyPromptShieldGuardrail,
)
from litellm.proxy.guardrails.guardrail_hooks.azure.text_moderation import (
    AzureContentSafetyTextModerationGuardrail,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import CallTypesLiteral


class TupleWriter(CustomGuardrail):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object]:
        messages: Final = data.get("messages")
        if isinstance(messages, list):
            data["messages"] = tuple(messages)  # mutable-ok: rewrite messages for the dispatch regression
        return data


class AllTurnsPromptShield(AzureContentSafetyPromptShieldGuardrail):
    def get_user_prompt(self, messages: list[AllMessageValues]) -> str:
        return "\\n".join(
            message["content"]
            for message in messages
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        )


class AllTurnsTextModeration(AzureContentSafetyTextModerationGuardrail):
    def get_user_prompt(self, messages: list[AllMessageValues]) -> str:
        return "\\n".join(
            message["content"]
            for message in messages
            if isinstance(message, dict)
            and message.get("role") == "user"
            and isinstance(message.get("content"), str)
        )


class RequiringPromptShield(AzureContentSafetyPromptShieldGuardrail):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object] | None:
        messages: Final = data.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="no user text")
        user_prompt: Final = self.get_user_prompt(cast(list[AllMessageValues], messages))  # cast-ok: chat messages
        if not user_prompt:
            raise HTTPException(status_code=400, detail="no user text")
        return await super().async_pre_call_hook(user_api_key_dict, cache, data, call_type)


class RequiringTextModeration(AzureContentSafetyTextModerationGuardrail):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: CallTypesLiteral,
    ) -> dict[str, object] | None:
        messages: Final = data.get("messages")
        if not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="no user text")
        user_prompt: Final = self.get_user_prompt(cast(list[AllMessageValues], messages))  # cast-ok: chat messages
        if not user_prompt:
            raise HTTPException(status_code=400, detail="no user text")
        return await super().async_pre_call_hook(user_api_key_dict, cache, data, call_type)
"""


def _azure_text(request: Request) -> str:
    body: Final = object_value(json.loads(request.body))
    if request.target.startswith(_SHIELD_TARGET_PREFIX):
        shield_prompt: Final = body["userPrompt"]
        assert isinstance(shield_prompt, str), body
        return shield_prompt
    assert request.target.startswith(_ANALYZE_TARGET_PREFIX), request.target
    moderation_text: Final = body["text"]
    assert isinstance(moderation_text, str), body
    return moderation_text


def _azure(request: Request) -> Reply:
    assert request.method == "POST", request.method
    body: Final = object_value(json.loads(request.body))
    if request.target.startswith(_SHIELD_TARGET_PREFIX):
        prompt: Final = body["userPrompt"]
        assert isinstance(prompt, str), body
        return Reply(
            body=json.dumps(
                {
                    "userPromptAnalysis": {"attackDetected": _ATTACK_MARKER in prompt},
                    "documentsAnalysis": [],
                }
            ).encode()
        )
    assert request.target.startswith(_ANALYZE_TARGET_PREFIX), request.target
    text: Final = body["text"]
    assert isinstance(text, str), body
    severity: Final = 4 if _MODERATION_MARKER in text else 0
    return Reply(
        body=json.dumps(
            {
                "blocklistsMatch": [],
                "categoriesAnalysis": [
                    {"category": "Hate", "severity": severity},
                    {"category": "Sexual", "severity": 0},
                    {"category": "SelfHarm", "severity": 0},
                    {"category": "Violence", "severity": 0},
                ],
            }
        ).encode()
    )


def _provider(request: Request) -> Reply:
    assert request.method == "POST", request.method
    if request.target == "/v1/embeddings":
        return Reply(
            body=json.dumps(
                {
                    "object": "list",
                    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                    "model": "text-embedding-3-small",
                    "usage": {"prompt_tokens": 1, "total_tokens": 1},
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
                    "model": "gpt-4o-mini",
                    "output": [
                        {
                            "type": "message",
                            "id": "msg_" + uuid.uuid4().hex,
                            "status": "completed",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": "permitted response", "annotations": []}],
                        }
                    ],
                    "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
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
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "permitted response"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
        ).encode()
    )


def _guardrails(azure: Wire) -> list[dict[str, JsonValue]]:
    return [
        {
            "guardrail_name": "tuple-writer",
            "litellm_params": {
                "guardrail": "azure_dispatch_hooks.TupleWriter",
                "mode": "pre_call",
                "default_on": False,
            },
        },
        {
            "guardrail_name": "shield",
            "litellm_params": {
                "guardrail": "azure/prompt_shield",
                "mode": "pre_call",
                "default_on": False,
                "api_base": azure.url,
                "api_key": "synthetic-azure-key",
            },
        },
        {
            "guardrail_name": "moderation",
            "litellm_params": {
                "guardrail": "azure/text_moderations",
                "mode": "pre_call",
                "default_on": False,
                "api_base": azure.url,
                "api_key": "synthetic-azure-key",
            },
        },
        {
            "guardrail_name": "all-turns-shield",
            "litellm_params": {
                "guardrail": "azure_dispatch_hooks.AllTurnsPromptShield",
                "mode": "pre_call",
                "default_on": False,
                "api_base": azure.url,
                "api_key": "synthetic-azure-key",
            },
        },
        {
            "guardrail_name": "all-turns-moderation",
            "litellm_params": {
                "guardrail": "azure_dispatch_hooks.AllTurnsTextModeration",
                "mode": "pre_call",
                "default_on": False,
                "api_base": azure.url,
                "api_key": "synthetic-azure-key",
            },
        },
        {
            "guardrail_name": "requiring-shield",
            "litellm_params": {
                "guardrail": "azure_dispatch_hooks.RequiringPromptShield",
                "mode": "pre_call",
                "default_on": False,
                "api_base": azure.url,
                "api_key": "synthetic-azure-key",
            },
        },
        {
            "guardrail_name": "requiring-moderation",
            "litellm_params": {
                "guardrail": "azure_dispatch_hooks.RequiringTextModeration",
                "mode": "pre_call",
                "default_on": False,
                "api_base": azure.url,
                "api_key": "synthetic-azure-key",
            },
        },
    ]


@pytest.fixture(scope="module")
def azure_rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[tuple[Gateway, Wire, Wire]]:
    directory: Final = tmp_path_factory.mktemp("azure-content-safety-dispatch")
    with ExitStack() as stack:
        gateway: Final = stack.enter_context(gateway_from_environment())
        azure: Final = stack.enter_context(wire_server(_azure))
        provider: Final = stack.enter_context(wire_server(_provider))
        (directory / "azure_dispatch_hooks.py").write_text(_HOOKS_SOURCE)
        base_config: Final = yaml.safe_load(Path("tests/integration/proxy_config.yaml").read_text())
        general_settings: Final = {
            **base_config["general_settings"],
            "store_prompts_in_spend_logs": True,
        }
        config: Final = {
            **base_config,
            "guardrails": _guardrails(azure),
            "general_settings": general_settings,
        }
        config_path: Final = directory / "azure-content-safety-dispatch.yaml"
        config_path.write_text(yaml.safe_dump(config))
        candidate: Final = stack.enter_context(owned_proxy(gateway, directory, {}, config=config_path))
        yield candidate, azure, provider


@pytest.fixture(autouse=True)
def _clear_wires(azure_rig: tuple[Gateway, Wire, Wire]) -> None:
    azure_rig[1].drain()
    azure_rig[2].drain()


def _model(scenario: Scenario, provider: Wire, model: str = "openai/gpt-4o-mini") -> str:
    return scenario.model(
        model=model,
        api_base=provider.url + "/v1",
        api_key="synthetic-provider-key",
    )


def _guardrail_entry(model: str) -> dict[str, JsonValue]:
    rows: Final = eventually(
        lambda: read_rows('SELECT metadata FROM "LiteLLM_SpendLogs" WHERE model_group=%s', (model,)),
        lambda values: len(values) == 1,
        seconds=70,
    )
    metadata: Final = object_value(rows[0]["metadata"])
    entries: Final = metadata["guardrail_information"]
    assert isinstance(entries, list) and len(entries) == 1, metadata
    return object_value(entries[0])


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", _ATTACK_MARKER), ("moderation", _MODERATION_MARKER)],
)
def test_tuple_messages_are_scanned_for_each_guardrail(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"synthetic prompt {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "guardrails": ["tuple-writer", guardrail_name],
            },
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (prompt,)
        assert provider.drain() == ()


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("shield", _ATTACK_MARKER), ("moderation", _MODERATION_MARKER)],
)
def test_list_messages_are_scanned_for_each_guardrail(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"synthetic prompt {marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "guardrails": [guardrail_name],
            },
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (prompt,)
        assert provider.drain() == ()


def test_prompt_shield_scans_responses_input(azure_rig: tuple[Gateway, Wire, Wire]) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"synthetic prompt {_ATTACK_MARKER} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/responses",
            {"model": model, "input": prompt, "guardrails": ["shield"]},
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (prompt,)
        assert provider.drain() == ()


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("all-turns-shield", _ATTACK_MARKER), ("all-turns-moderation", _MODERATION_MARKER)],
)
def test_guardrail_subclass_prompt_override_scans_every_user_turn(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    first_prompt: Final = f"synthetic prompt {marker} {uuid.uuid4().hex}"
    expected_prompt: Final = first_prompt + "\nbenign final user turn"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [
                    {"role": "user", "content": first_prompt},
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "benign final user turn"},
                ],
                "guardrails": [guardrail_name],
            },
        )
        assert response.status_code == 400, response.text
        assert _azure_texts(azure) == (expected_prompt,)
        assert provider.drain() == ()


@pytest.mark.parametrize(
    ("guardrail_name", "marker"),
    [("requiring-shield", "benign shield prompt"), ("requiring-moderation", "benign moderation prompt")],
)
def test_guardrail_subclass_can_call_get_user_prompt(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str, marker: str
) -> None:
    candidate, azure, provider = azure_rig
    prompt: Final = f"{marker} {uuid.uuid4().hex}"
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider)
        response: Final = candidate.request(
            "POST",
            "/v1/chat/completions",
            {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "guardrails": [guardrail_name],
            },
        )
        assert response.status_code == 200, response.text
        assert "permitted response" in response.text
        assert _azure_texts(azure) == (prompt,)
        assert len(provider.drain()) == 1


@pytest.mark.parametrize("guardrail_name", ["shield", "moderation"])
def test_messages_less_embeddings_log_allow_without_azure_request(
    azure_rig: tuple[Gateway, Wire, Wire], guardrail_name: str
) -> None:
    candidate, azure, provider = azure_rig
    with candidate.scenario() as scenario:
        model: Final = _model(scenario, provider, model="openai/text-embedding-3-small")
        response: Final = candidate.request(
            "POST",
            "/v1/embeddings",
            {"model": model, "input": "synthetic benign embedding text", "guardrails": [guardrail_name]},
        )
        assert response.status_code == 200, response.text
        assert _azure_texts(azure) == ()
        assert len(provider.drain()) == 1
        entry: Final = _guardrail_entry(model)
        assert entry["guardrail_status"] == "success", entry
        assert entry["guardrail_response"] == "allow", entry


def _azure_texts(azure: Wire) -> tuple[str, ...]:
    return tuple(_azure_text(request) for request in azure.drain())
