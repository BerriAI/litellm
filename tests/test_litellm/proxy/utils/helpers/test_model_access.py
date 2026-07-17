from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

import litellm
from litellm import ModelResponse
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import (
    create_model_info_response,
    get_available_models_for_user,
    is_known_model,
    is_known_vector_store_index,
    model_dump_with_preserved_fields,
    validate_model_access,
)


def normalize(value):
    return value


def _router_with_models(model_names):
    router = MagicMock()
    router.get_model_names.return_value = model_names
    router.get_model_access_groups.return_value = {}
    return router


def test_is_known_model_happy_path_returns_true_when_in_router():
    router = _router_with_models(["gpt-4o", "claude-haiku"])
    summary = {
        "result": is_known_model("gpt-4o", router),
        "model": "gpt-4o",
        "router_models": ["gpt-4o", "claude-haiku"],
    }
    assert summary == {
        "result": True,
        "model": "gpt-4o",
        "router_models": ["gpt-4o", "claude-haiku"],
    }


def test_is_known_model_returns_false_when_not_in_router():
    router = _router_with_models(["gpt-4o"])
    summary = {
        "result": is_known_model("claude-haiku", router),
        "model": "claude-haiku",
        "router_models": ["gpt-4o"],
    }
    assert summary == {
        "result": False,
        "model": "claude-haiku",
        "router_models": ["gpt-4o"],
    }


def test_is_known_model_error_path_none_model():
    router = _router_with_models(["gpt-4o"])
    assert is_known_model(None, router) is False


def test_is_known_model_error_path_none_router():
    assert is_known_model("gpt-4o", None) is False


def test_is_known_vector_store_index_happy_path(monkeypatch):
    registry = MagicMock()
    registry.get_vector_store_indexes.return_value = ["index-a", "index-b"]
    monkeypatch.setattr(litellm, "vector_store_index_registry", registry)
    summary = {
        "result": is_known_vector_store_index("index-a"),
        "indexes": ["index-a", "index-b"],
        "input": "index-a",
    }
    assert summary == {
        "result": True,
        "indexes": ["index-a", "index-b"],
        "input": "index-a",
    }


def test_is_known_vector_store_index_returns_false_when_missing(monkeypatch):
    registry = MagicMock()
    registry.get_vector_store_indexes.return_value = ["index-a"]
    monkeypatch.setattr(litellm, "vector_store_index_registry", registry)
    summary = {
        "result": is_known_vector_store_index("missing"),
        "indexes": ["index-a"],
        "input": "missing",
    }
    assert summary == {
        "result": False,
        "indexes": ["index-a"],
        "input": "missing",
    }


def test_is_known_vector_store_index_error_path_no_registry(monkeypatch):
    monkeypatch.setattr(litellm, "vector_store_index_registry", None)
    assert is_known_vector_store_index("anything") is False


def test_create_model_info_response_happy_path_no_metadata():
    result = create_model_info_response(model_id="gpt-4o", provider="openai")
    assert result == {
        "id": "gpt-4o",
        "object": "model",
        "created": result["created"],
        "owned_by": "openai",
        "max_input_tokens": 128000,
        "max_output_tokens": 16384,
    }
    snapshot = {
        "id": result["id"],
        "object": result["object"],
        "owned_by": result["owned_by"],
        "created_is_int": isinstance(result["created"], int),
        "metadata_absent": "metadata" not in result,
    }
    assert snapshot == {
        "id": "gpt-4o",
        "object": "model",
        "owned_by": "openai",
        "created_is_int": True,
        "metadata_absent": True,
    }


def test_create_model_info_response_with_metadata_default_general(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_all_fallbacks",
        lambda **_kwargs: [{"model": "fallback-1"}],
    )
    result = create_model_info_response(
        model_id="gpt-4o",
        provider="openai",
        include_metadata=True,
    )
    snapshot = {
        "id": result["id"],
        "owned_by": result["owned_by"],
        "object": result["object"],
        "fallbacks": result["metadata"]["fallbacks"],
    }
    assert snapshot == {
        "id": "gpt-4o",
        "owned_by": "openai",
        "object": "model",
        "fallbacks": [{"model": "fallback-1"}],
    }


def test_create_model_info_response_with_explicit_fallback_type(monkeypatch):
    captured = {}

    def _capture(model, llm_router, fallback_type):
        captured["fallback_type"] = fallback_type
        return ["x"]

    monkeypatch.setattr("litellm.proxy.auth.model_checks.get_all_fallbacks", _capture)
    result = create_model_info_response(
        model_id="gpt-4o",
        provider="openai",
        include_metadata=True,
        fallback_type="context_window",
    )
    snapshot = {
        "id": result["id"],
        "fallbacks": result["metadata"]["fallbacks"],
        "captured_fallback_type": captured["fallback_type"],
        "owned_by": result["owned_by"],
    }
    assert snapshot == {
        "id": "gpt-4o",
        "fallbacks": ["x"],
        "captured_fallback_type": "context_window",
        "owned_by": "openai",
    }


def test_create_model_info_response_invalid_fallback_type_raises():
    with pytest.raises(HTTPException) as exc_info:
        create_model_info_response(
            model_id="gpt-4o",
            provider="openai",
            include_metadata=True,
            fallback_type="bogus",
        )
    assert exc_info.value.status_code == 400
    assert "Invalid fallback_type" in str(exc_info.value.detail)


def test_validate_model_access_happy_path_single_model_in_list():
    summary = {
        "result": validate_model_access("gpt-4o", ["gpt-4o", "claude-haiku"]),
        "model": "gpt-4o",
        "available": ["gpt-4o", "claude-haiku"],
    }
    assert summary == {
        "result": None,
        "model": "gpt-4o",
        "available": ["gpt-4o", "claude-haiku"],
    }


def test_validate_model_access_happy_path_batch_all_accessible():
    summary = {
        "result": validate_model_access(
            "gpt-4o,claude-haiku", ["gpt-4o", "claude-haiku", "gemini"]
        ),
        "input": "gpt-4o,claude-haiku",
        "available": ["gpt-4o", "claude-haiku", "gemini"],
    }
    assert summary == {
        "result": None,
        "input": "gpt-4o,claude-haiku",
        "available": ["gpt-4o", "claude-haiku", "gemini"],
    }


def test_validate_model_access_single_model_not_accessible_raises():
    with pytest.raises(HTTPException) as exc_info:
        validate_model_access("missing-model", ["gpt-4o"])
    assert exc_info.value.status_code == 404
    assert "missing-model" in str(exc_info.value.detail)


def test_validate_model_access_batch_partial_inaccessible_raises():
    with pytest.raises(HTTPException) as exc_info:
        validate_model_access("gpt-4o,unknown-x", ["gpt-4o"])
    assert exc_info.value.status_code == 404
    assert "unknown-x" in str(exc_info.value.detail)
    assert "gpt-4o" not in str(exc_info.value.detail).split("not accessible:")[1]


def _make_model_response():
    return ModelResponse(
        id="resp-123",
        choices=[
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "do_thing", "arguments": "{}"},
                        }
                    ],
                },
                "index": 0,
                "finish_reason": "tool_calls",
            }
        ],
        model="gpt-4o",
    )


def test_model_dump_with_preserved_fields_restores_none_content():
    resp = _make_model_response()
    result = model_dump_with_preserved_fields(resp)
    message = result["choices"][0]["message"]
    snapshot = {
        "content_is_none": message["content"] is None,
        "role": message["role"],
        "has_tool_calls": "tool_calls" in message,
        "model": result["model"],
    }
    assert snapshot == {
        "content_is_none": True,
        "role": "assistant",
        "has_tool_calls": True,
        "model": "gpt-4o",
    }


def test_model_dump_with_preserved_fields_no_choices_returns_plain_dump():
    class _Bare:
        def model_dump(self, **_kwargs):
            return {"id": "x", "object": "y", "extra": "z"}

    bare = _Bare()
    result = model_dump_with_preserved_fields(bare)
    assert result == {"id": "x", "object": "y", "extra": "z"}


def test_model_dump_with_preserved_fields_error_path_invalid_obj_raises():
    with pytest.raises(AttributeError):
        model_dump_with_preserved_fields(None)


@pytest.mark.asyncio
async def test_get_available_models_for_user_happy_path_returns_complete_list(
    monkeypatch,
):
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_key_models",
        lambda **_k: ["gpt-4o"],
    )
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_team_models",
        lambda **_k: ["claude-haiku"],
    )
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_complete_model_list",
        lambda **_k: ["gpt-4o", "claude-haiku", "gemini"],
    )
    router = _router_with_models(["gpt-4o", "claude-haiku", "gemini"])
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key",
        user_id="user-1",
        team_id=None,
        team_models=[],
    )
    result = await get_available_models_for_user(
        user_api_key_dict=user_api_key_dict,
        llm_router=router,
        general_settings={},
        user_model=None,
    )
    summary = {
        "result_sorted": sorted(result),
        "count": len(result),
        "user_id": user_api_key_dict.user_id,
        "router_set": True,
    }
    assert summary == {
        "result_sorted": ["claude-haiku", "gemini", "gpt-4o"],
        "count": 3,
        "user_id": "user-1",
        "router_set": True,
    }


@pytest.mark.asyncio
async def test_get_available_models_for_user_with_none_router(monkeypatch):
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_key_models",
        lambda **_k: [],
    )
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_team_models",
        lambda **_k: [],
    )
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_complete_model_list",
        lambda **_k: ["user-model"],
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key",
        user_id="user-1",
        team_id=None,
        team_models=[],
    )
    result = await get_available_models_for_user(
        user_api_key_dict=user_api_key_dict,
        llm_router=None,
        general_settings={},
        user_model="user-model",
    )
    summary = {
        "result": result,
        "router_is_none": True,
        "user_model": "user-model",
        "count": len(result),
    }
    assert summary == {
        "result": ["user-model"],
        "router_is_none": True,
        "user_model": "user-model",
        "count": 1,
    }


@pytest.mark.asyncio
async def test_get_available_models_for_user_error_path_complete_list_raises(
    monkeypatch,
):
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_key_models",
        lambda **_k: [],
    )
    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_team_models",
        lambda **_k: [],
    )

    def _boom(**_kwargs):
        raise RuntimeError("downstream failure")

    monkeypatch.setattr(
        "litellm.proxy.auth.model_checks.get_complete_model_list", _boom
    )
    user_api_key_dict = UserAPIKeyAuth(
        api_key="sk-test-key",
        user_id="user-1",
        team_id=None,
        team_models=[],
    )
    with pytest.raises(RuntimeError):
        await get_available_models_for_user(
            user_api_key_dict=user_api_key_dict,
            llm_router=None,
            general_settings={},
            user_model=None,
        )
