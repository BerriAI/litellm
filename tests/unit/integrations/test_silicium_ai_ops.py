# SPDX-License-Identifier: MIT
"""Unit tests for LiteLLM `SiliciumAiOpsLogger` integration (`v1.108+`)."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
from collections.abc import Mapping
from typing import Final

import pytest


def _load_logger_cls() -> type:
    for mod_name in ("litellm.integrations.silicium_ai_ops", "litellm_silicium_integration"):
        with contextlib.suppress(ImportError, AttributeError):
            mod: Final = importlib.import_module(mod_name)
            cls: Final = getattr(mod, "SiliciumAiOpsLogger", None)
            if isinstance(cls, type):
                return cls
    raise RuntimeError("Unable to import SiliciumAiOpsLogger")


class _FakeGateway:
    def __init__(self, pii_policy: str = "mask") -> None:
        self.pii_policy: Final = pii_policy
        self.cached_entries: int = 0
        self.last_logged_payload: Mapping[str, object] = {}

    def __call__(self, *args: object) -> object:
        if not args:
            return self
        return {"called_args": len(args)}

    def pre_call_hook(
        self,
        request_payload: object = None,
        raise_on_block: bool = False,
        data: object = None,
        call_type: object = None,
    ) -> Mapping[str, object]:
        target: Final = data if data is not None else request_payload
        raw_str: Final = str(target)
        has_pii: Final = "ceo@example.com" in raw_str
        pii_flags: Final = 1 if has_pii else 0
        is_hit: Final = self.cached_entries > 0
        insp_summary: Final = {
            "allowed": not (raise_on_block and has_pii),
            "cache_hit": is_hit,
            "kv_prefix_hash": "a1b2c3d4e5f60718",
            "worker_id": 1,
            "estimated_tokens": 1,
            "pii_flags": pii_flags,
            "headers": {
                "x-silicium-cache-hit": "true" if is_hit else "false",
                "x-silicium-kv-prefix-hash": "a1b2c3d4e5f60718",
                "x-silicium-worker-id": "1",
            },
        }
        if isinstance(data, Mapping) and call_type is not None:
            redacted_text: Final = (
                raw_str.replace("ceo@example.com", "[SILICIUM_REDACTED_EMAIL]")
                if isinstance(data.get("prompt"), str)
                else "[SILICIUM_REDACTED_EMAIL]"
            )
            return {
                "model": "mutated-model-should-be-ignored",
                "messages": [{"role": "user", "content": "[SILICIUM_REDACTED_EMAIL]"}],
                "prompt": redacted_text,
                "metadata": {"silicium_inspection": insp_summary},
            }
        return {
            "allowed": not (raise_on_block and has_pii),
            "cache_hit": is_hit,
            "kv_prefix_hash": "a1b2c3d4e5f60718",
            "worker_id": 1,
            "estimated_tokens": 1,
            "pii_flags": pii_flags,
            "sanitized_payload_str": "[SILICIUM_REDACTED_EMAIL]",
        }

    async def async_pre_call_hook(
        self,
        request_payload: object = None,
        raise_on_block: bool = False,
        data: object = None,
        call_type: object = None,
    ) -> Mapping[str, object]:
        return self.pre_call_hook(
            request_payload=request_payload,
            raise_on_block=raise_on_block,
            data=data,
            call_type=call_type,
        )

    def post_call_hook(
        self,
        request_payload: object,
        response_obj: object = None,
        prompt: str = "",
        end_time: object = None,
    ) -> Mapping[str, object]:
        _ = response_obj
        if isinstance(request_payload, Mapping):
            self.last_logged_payload = dict(request_payload)
        self.cached_entries += 1
        return {"cached": True, "prompt": prompt, "end_time": end_time}

    async def async_log_success_event(
        self,
        kwargs: object,
        response_obj: object = None,
        prompt: str = "",
        end_time: object = None,
    ) -> Mapping[str, object]:
        return self.post_call_hook(
            kwargs,
            response_obj=response_obj,
            prompt=prompt,
            end_time=end_time,
        )


def test_silicium_ai_ops_logger_4arg_proxy_and_1arg_sdk_contract() -> None:
    logger_cls: Final = _load_logger_cls()
    default_logger: Final = logger_cls()
    assert default_logger.enforces_request_content is True
    assert default_logger.enable_semantic_cache is False

    with pytest.raises(ValueError, match="tenant_scope"):
        logger_cls(enable_semantic_cache=True)

    fake_gw: Final = _FakeGateway(pii_policy="mask")
    logger: Final = logger_cls(
        pii_policy="mask",
        enable_semantic_cache=True,
        tenant_scope="cluster-prod",
        delegate=fake_gw,
    )
    assert logger() is logger

    proxy_req: Final = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Email ceo@example.com"}],
        "metadata": {"user_api_key_team_id": "team-alpha", "user_api_key": "hashed-key-1"},
    }
    res_4arg: Final = asyncio.run(
        logger.async_pre_call_hook({"user_api_key_team_id": "team-alpha"}, object(), proxy_req, "completion")
    )
    assert res_4arg.get("model") == "gpt-4o-mini"
    messages_out: Final = res_4arg.get("messages")
    assert isinstance(messages_out, list)
    first_msg: Final = messages_out[0]
    assert isinstance(first_msg, dict)
    assert first_msg.get("content") == "[SILICIUM_REDACTED_EMAIL]"
    meta_out: Final = res_4arg.get("metadata")
    assert isinstance(meta_out, dict)
    assert meta_out.get("user_api_key_team_id") == "team-alpha"
    assert meta_out.get("user_api_key") == "hashed-key-1"
    insp_out: Final = meta_out.get("silicium_inspection")
    assert isinstance(insp_out, dict)
    assert insp_out.get("allowed") is True
    assert insp_out.get("cache_hit") is False
    expected_alpha_scope: Final = (
        "scope=cluster-prod|user_api_key_team_id=team-alpha|"
        "user_api_key_team_id=team-alpha|user_api_key=hashed-key-1|model=gpt-4o-mini"
    )
    assert insp_out.get("tenant_scope") == expected_alpha_scope
    assert insp_out.get("kv_prefix_hash") == "a1b2c3d4e5f60718"
    assert insp_out.get("worker_id") == 1

    structured_req: Final = {
        "model": "gpt-4.1",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Contact ceo@example.com"}],
            }
        ],
        "input": [
            {"type": "input_text", "text": "Reach ceo@example.com"},
        ],
        "metadata": {"user_api_key_team_id": "team-alpha"},
    }
    res_structured: Final = asyncio.run(
        logger.async_pre_call_hook({"user_api_key_team_id": "team-alpha"}, object(), structured_req, "responses")
    )
    struct_msgs: Final = res_structured.get("messages")
    assert isinstance(struct_msgs, list)
    struct_blocks: Final = struct_msgs[0].get("content")
    assert isinstance(struct_blocks, list)
    assert struct_blocks[0].get("text") == "[SILICIUM_REDACTED_EMAIL]"
    struct_input: Final = res_structured.get("input")
    assert isinstance(struct_input, list)
    assert struct_input[0].get("text") == "[SILICIUM_REDACTED_EMAIL]"

    unsupported_req: Final = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": "https://x.test/ceo@example.com"}}],
            }
        ],
    }
    with pytest.raises(PermissionError):
        asyncio.run(
            logger.async_pre_call_hook({"user_api_key_team_id": "team-alpha"}, object(), unsupported_req, "completion")
        )

    raw_logging_kwargs: Final = {
        "api_key": "sk-secret-provider-credential-must-not-leak",
        "litellm_params": {
            "api_key": "sk-secret-provider-credential-must-not-leak",
            "metadata": {"user_api_key_team_id": "team-alpha", "user_api_key": "hashed-key-1"},
        },
        "standard_logging_object": {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": "[SILICIUM_REDACTED_EMAIL]"}],
            "call_type": "completion",
        },
    }
    log_out: Final = asyncio.run(
        logger.async_log_success_event(raw_logging_kwargs, {"id": "chatcmpl-1"}, 0.0, 0.02)
    )
    assert log_out.get("cached") is True
    assert "api_key" not in fake_gw.last_logged_payload
    assert "litellm_params" not in fake_gw.last_logged_payload

    res_other_tenant: Final = asyncio.run(
        logger.async_pre_call_hook(
            {"user_api_key_team_id": "team-beta"},
            object(),
            {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": "Clean question"}],
                "metadata": {"user_api_key_team_id": "team-beta"},
            },
            "completion",
        )
    )
    other_meta: Final = res_other_tenant.get("metadata")
    assert isinstance(other_meta, dict)
    other_insp: Final = other_meta.get("silicium_inspection")
    assert isinstance(other_insp, dict)
    assert other_insp.get("cache_hit") is False

    sdk_req: Final = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Email ceo@example.com"}],
    }
    _ = asyncio.run(logger.async_log_success_event(sdk_req, {"id": "chatcmpl-2"}, 0.0, 0.01))
    res_1arg: Final = asyncio.run(logger.async_pre_call_hook(sdk_req))
    assert res_1arg.get("allowed") is True
    assert res_1arg.get("cache_hit") is True
    assert res_1arg.get("tenant_scope") == "scope=cluster-prod|model=gpt-4o-mini"
    assert res_1arg.get("kv_prefix_hash") == "a1b2c3d4e5f60718"
    assert res_1arg.get("sanitized_payload_str") == "[SILICIUM_REDACTED_EMAIL]"

    blocking_logger: Final = logger_cls(pii_policy="block", delegate=_FakeGateway(pii_policy="block"))
    with pytest.raises(PermissionError):
        asyncio.run(
            blocking_logger.async_pre_call_hook({"api_key": "k"}, object(), proxy_req, "completion")
        )