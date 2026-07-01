from typing import Optional
from unittest.mock import patch

import pytest

import litellm
from litellm.llms.litellm_proxy.chat.transformation import LiteLLMProxyChatConfig

MESSAGES = [{"role": "user", "content": "Hello"}]


def _sync_transform(
    config: LiteLLMProxyChatConfig,
    model: str = "gpt-4o",
    messages: list = MESSAGES,
    optional_params: Optional[dict] = None,
    litellm_params: Optional[dict] = None,
) -> dict:
    return config.transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params or {},
        litellm_params=litellm_params or {},
        headers={},
    )


async def _async_transform(
    config: LiteLLMProxyChatConfig,
    model: str = "gpt-4o",
    messages: list = MESSAGES,
    optional_params: Optional[dict] = None,
    litellm_params: Optional[dict] = None,
) -> dict:
    return await config.async_transform_request(
        model=model,
        messages=messages,
        optional_params=optional_params or {},
        litellm_params=litellm_params or {},
        headers={},
    )


def test_litellm_proxy_chat_transformation():
    """
    Assert messages are not transformed when calling litellm proxy
    """
    config = LiteLLMProxyChatConfig()
    file_content = [
        {"type": "text", "text": "What is this document about?"},
        {
            "type": "file",
            "file": {
                "file_id": "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf",
                "format": "application/pdf",
            },
        },
    ]
    messages = [{"role": "user", "content": file_content}]
    assert config.transform_request(
        model="model",
        messages=messages,
        optional_params={},
        litellm_params={},
        headers={},
    ) == {"model": "model", "messages": messages}


def test_litellm_gateway_from_sdk_with_user_param():
    from litellm.llms.litellm_proxy.chat.transformation import LiteLLMProxyChatConfig

    supported_params = LiteLLMProxyChatConfig().get_supported_openai_params("openai/gpt-4o")
    print(f"supported_params: {supported_params}")
    assert "user" in supported_params


# --- tags forwarding from requester_metadata ---


def test_tags_forwarded_from_requester_metadata():
    config = LiteLLMProxyChatConfig()
    litellm_params = {"metadata": {"requester_metadata": {"tags": ["project-x", "cost-center-42"]}}}
    body = _sync_transform(config, litellm_params=litellm_params)
    assert body["metadata"] == {"tags": ["project-x", "cost-center-42"]}


def test_non_allowlisted_fields_not_forwarded():
    config = LiteLLMProxyChatConfig()
    litellm_params = {
        "metadata": {
            "requester_metadata": {
                "tags": ["ok"],
                "user_api_key_hash": "secret",
                "guardrails_config": {"block": True},
                "spend_logs_metadata": {"internal": True},
            }
        }
    }
    body = _sync_transform(config, litellm_params=litellm_params)
    assert body["metadata"] == {"tags": ["ok"]}
    assert "user_api_key_hash" not in body["metadata"]
    assert "guardrails_config" not in body["metadata"]
    assert "spend_logs_metadata" not in body["metadata"]


def test_no_metadata_key_when_no_allowlisted_fields():
    config = LiteLLMProxyChatConfig()
    litellm_params = {"metadata": {"requester_metadata": {"user_api_key_hash": "secret"}}}
    body = _sync_transform(config, litellm_params=litellm_params)
    assert "metadata" not in body


def test_no_metadata_key_when_requester_metadata_absent():
    config = LiteLLMProxyChatConfig()
    litellm_params = {"metadata": {"other_key": "value"}}
    body = _sync_transform(config, litellm_params=litellm_params)
    assert "metadata" not in body


def test_no_metadata_key_when_metadata_absent():
    config = LiteLLMProxyChatConfig()
    body = _sync_transform(config, litellm_params={})
    assert "metadata" not in body


# --- litellm_session_id forwarding ---


def test_session_id_injected_into_extra_body():
    config = LiteLLMProxyChatConfig()
    litellm_params = {"litellm_session_id": "session-abc-123"}
    body = _sync_transform(config, litellm_params=litellm_params)
    assert body["extra_body"]["litellm_session_id"] == "session-abc-123"


def test_session_id_does_not_overwrite_existing_extra_body_value():
    config = LiteLLMProxyChatConfig()
    litellm_params = {"litellm_session_id": "upstream-session"}
    optional_params = {"extra_body": {"litellm_session_id": "caller-set-session"}}
    body = _sync_transform(config, optional_params=optional_params, litellm_params=litellm_params)
    assert body["extra_body"]["litellm_session_id"] == "caller-set-session"


def test_session_id_absent_when_not_in_litellm_params():
    config = LiteLLMProxyChatConfig()
    body = _sync_transform(config, litellm_params={})
    assert "extra_body" not in body or "litellm_session_id" not in body.get("extra_body", {})


# --- combined tags + session_id ---


def test_tags_and_session_id_both_forwarded():
    config = LiteLLMProxyChatConfig()
    litellm_params = {
        "metadata": {"requester_metadata": {"tags": ["sprint-3"]}},
        "litellm_session_id": "session-xyz",
    }
    body = _sync_transform(config, litellm_params=litellm_params)
    assert body["metadata"] == {"tags": ["sprint-3"]}
    assert body["extra_body"]["litellm_session_id"] == "session-xyz"


# --- base fields regression ---


def test_optional_params_passed_through():
    config = LiteLLMProxyChatConfig()
    body = _sync_transform(config, optional_params={"temperature": 0.7, "max_tokens": 100})
    assert body["temperature"] == 0.7
    assert body["max_tokens"] == 100


# --- async path matches sync ---


@pytest.mark.asyncio
async def test_async_matches_sync_for_tags_and_session():
    config = LiteLLMProxyChatConfig()
    litellm_params = {
        "metadata": {"requester_metadata": {"tags": ["async-test"]}},
        "litellm_session_id": "async-session-999",
    }
    sync_body = _sync_transform(config, litellm_params=litellm_params)
    async_body = await _async_transform(config, litellm_params=litellm_params)
    assert sync_body == async_body


@pytest.mark.asyncio
async def test_async_no_metadata_when_empty():
    config = LiteLLMProxyChatConfig()
    async_body = await _async_transform(config, litellm_params={})
    assert "metadata" not in async_body
    assert "extra_body" not in async_body
