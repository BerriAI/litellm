"""Regression tests: when a key is denied model access, the proxy must log which
key/user was denied which model (server-side warning) — while the client-facing
exception stays free of key identity, and successful access-group fallbacks are
not miscounted as denials."""

import logging
from unittest.mock import AsyncMock, patch

import pytest
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_checks import can_key_call_model

_LLM_MODEL_LIST = [
    {
        "model_name": "openai/*",
        "litellm_params": {"model": "openai/*", "api_key": "test-api-key"},
        "model_info": {
            "id": "e6e7006f83029df40ebc02ddd068890253f4cd3092bcb203d3d8e6f6f606f30f",
            "db_model": False,
            "access_groups": ["public-openai-models"],
        },
    },
    {
        "model_name": "openai/gpt-4o",
        "litellm_params": {"model": "openai/gpt-4o", "api_key": "test-api-key"},
        "model_info": {
            "id": "0cfcd87f2cb12a783a466888d05c6c89df66db23e01cecd75ec0b83aed73c9ad",
            "db_model": False,
            "access_groups": ["private-openai-models"],
        },
    },
]


def _auth(**overrides) -> UserAPIKeyAuth:
    kwargs = {
        "models": ["public-openai-models"],
        "key_alias": "prod-key",
        "key_name": "prod",
        "user_id": "user-123",
    }
    kwargs.update(overrides)
    return UserAPIKeyAuth(**kwargs)


def _denial_records(caplog) -> list:
    return [r for r in caplog.records if "Model access denied" in r.getMessage()]


@pytest.mark.asyncio
async def test_denial_without_fallback_logs_key_context(caplog):
    """A key denied natively with no usable access-group fallback must log the
    key identity + requested/allowed models server-side."""
    import litellm

    router = litellm.Router(model_list=_LLM_MODEL_LIST)
    caplog.set_level(logging.WARNING, logger="LiteLLM Proxy")

    with pytest.raises(Exception, match="is not available for this API key") as exc_info:
        await can_key_call_model(
            model="openai/gpt-4o",
            llm_model_list=_LLM_MODEL_LIST,
            valid_token=_auth(),
            llm_router=router,
        )

    records = _denial_records(caplog)
    assert records, "denial must produce a server-side Model access denied warning"
    message = records[-1].getMessage()
    assert "key_alias=prod-key" in message
    assert "key_name=prod" in message
    assert "user_id=user-123" in message
    assert "requested_model=openai/gpt-4o" in message
    assert "public-openai-models" in message
    assert "prod-key" not in str(exc_info.value), "client-facing message must stay free of key identity"


@pytest.mark.asyncio
async def test_denial_after_access_group_fallback_logs_key_context(caplog):
    """When the access-group fallback is attempted and also denies, the denial
    must still be logged exactly once."""
    import litellm

    router = litellm.Router(model_list=_LLM_MODEL_LIST)
    caplog.set_level(logging.WARNING, logger="LiteLLM Proxy")

    with patch(
        "litellm.proxy.auth.auth_checks.get_models_from_access_groups",
        new_callable=AsyncMock,
        return_value=["unrelated-model-group"],
    ):
        with pytest.raises(Exception, match="is not available for this API key"):
            await can_key_call_model(
                model="openai/gpt-4o",
                llm_model_list=_LLM_MODEL_LIST,
                valid_token=_auth(access_group_ids=["ag-1"]),
                llm_router=router,
            )

    records = _denial_records(caplog)
    assert len(records) == 1, "fallback denial path must log exactly one denial"
    assert "key_alias=prod-key" in records[0].getMessage()
    assert "unrelated-model-group" in records[0].getMessage()


@pytest.mark.asyncio
async def test_authorized_model_logs_no_denial(caplog):
    import litellm

    router = litellm.Router(model_list=_LLM_MODEL_LIST)
    caplog.set_level(logging.WARNING, logger="LiteLLM Proxy")

    await can_key_call_model(
        model="openai/gpt-4o-mini",
        llm_model_list=_LLM_MODEL_LIST,
        valid_token=_auth(models=["*"]),
        llm_router=router,
    )

    assert not _denial_records(caplog), "authorized request must not log a denial"
