import os
import sys
import pytest


sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi import HTTPException

from litellm.proxy.prompts.prompt_completion_service import (
    PromptCompletionService,
    PromptCompletionResponse,
)
from litellm.integrations.dotprompt import DotpromptManager
from litellm.integrations.gitlab import GitLabPromptManager
from litellm.proxy._types import LitellmUserRoles


@pytest.fixture
def mock_user_key():
    return MagicMock(
        user_id="u1",
        user_role=LitellmUserRoles.PROXY_ADMIN,
        metadata={"prompts": ["promptA", "id1"]},
    )


# -------------------------------------------------------------------
# Access control
# -------------------------------------------------------------------
def test_validate_access_allows_admin(mock_user_key):
    svc = PromptCompletionService(mock_user_key)
    svc.validate_access("promptA")  # Should not raise


def test_validate_access_blocks_missing_prompt(mock_user_key):
    svc = PromptCompletionService(mock_user_key)
    mock_user_key.metadata = {"prompts": ["other"]}
    with pytest.raises(HTTPException) as e:
        svc.validate_access("promptA")
    assert e.value.status_code == 400


def test_validate_access_blocks_wrong_role(mock_user_key):
    mock_user_key.user_role = "USER"
    svc = PromptCompletionService(mock_user_key)
    with pytest.raises(HTTPException) as e:
        svc.validate_access("promptA")
    assert e.value.status_code == 403


# -------------------------------------------------------------------
# Prompt loading
# -------------------------------------------------------------------
@patch("litellm.proxy.prompts.prompt_completion_service.PROMPT_HUB")
def test_load_prompt_and_callback_missing_prompt(mock_hub, mock_user_key):
    mock_hub.get_prompt_by_id.return_value = None
    svc = PromptCompletionService(mock_user_key)
    with pytest.raises(HTTPException) as e:
        svc.load_prompt_and_callback("notfound")
    assert e.value.status_code == 404


@patch("litellm.proxy.prompts.prompt_completion_service.PROMPT_HUB")
def test_load_prompt_and_callback_no_callback(mock_hub, mock_user_key):
    mock_hub.get_prompt_by_id.return_value = MagicMock()
    mock_hub.get_prompt_callback_by_id.return_value = None
    svc = PromptCompletionService(mock_user_key)
    with pytest.raises(HTTPException) as e:
        svc.load_prompt_and_callback("promptA")
    assert e.value.status_code == 404


@patch("litellm.proxy.prompts.prompt_completion_service.PROMPT_HUB")
def test_load_prompt_and_callback_dotprompt(mock_hub, mock_user_key):
    mock_prompt_spec = MagicMock()
    mock_prompt_spec.model_dump.return_value = {"prompt_id": "promptA"}
    mock_hub.get_prompt_by_id.return_value = mock_prompt_spec

    mock_dot = MagicMock(spec=DotpromptManager)
    mock_dot.prompt_manager.get_all_prompts_as_json.return_value = {
        "promptA": {"content": "Hello", "metadata": {"model": "gpt-4"}}
    }
    mock_hub.get_prompt_callback_by_id.return_value = mock_dot

    svc = PromptCompletionService(mock_user_key)
    cb, tmpl, spec = svc.load_prompt_and_callback("promptA")

    assert tmpl.content == "Hello"
    assert tmpl.metadata["model"] == "gpt-4"

@patch("litellm.proxy.prompts.prompt_completion_service.PROMPT_HUB")
def test_load_prompt_and_callback_gitlab(mock_hub, mock_user_key):
    mock_prompt_spec = MagicMock()
    mock_prompt_spec.model_dump.return_value = {
        "prompt_id": "id1",
        "litellm_params": {
            "model_config": {
                "content": "Hi",
                "metadata": {"model": "gpt-4"},
            }
        },
    }
    mock_hub.get_prompt_by_id.return_value = mock_prompt_spec

    mock_gl = MagicMock(spec=GitLabPromptManager)
    mock_hub.get_prompt_callback_by_id.return_value = mock_gl

    svc = PromptCompletionService(mock_user_key)
    cb, tmpl, spec = svc.load_prompt_and_callback("id1")

    assert tmpl.content == "Hi"
    assert tmpl.metadata["model"] == "gpt-4"
    assert cb is mock_gl


# -------------------------------------------------------------------
# Param merging and flattening
# -------------------------------------------------------------------
def test_flatten_and_merge_params(mock_user_key):
    svc = PromptCompletionService(mock_user_key)
    metadata = {"config": {"temperature": 0.5, "model": "bad"}}
    prompt_spec = MagicMock(litellm_params={"config": {"max_tokens": 100}})
    user_overrides = {"config": {"temperature": 0.9, "stream": True}}

    merged = svc.merge_params(metadata, prompt_spec, user_overrides)
    assert merged["temperature"] == 0.9
    assert merged["max_tokens"] == 100
    assert merged["user"] == "u1"
    assert "model" not in merged


# -------------------------------------------------------------------
# Model execution
# -------------------------------------------------------------------
@patch("litellm.acompletion", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_execute_model_completion_invokes_litellm(mock_acompletion, mock_user_key):
    svc = PromptCompletionService(mock_user_key)
    mock_acompletion.return_value = {"choices": []}

    result = await svc.execute_model_completion("gpt-4", ("gpt-4", []), {})
    mock_acompletion.assert_awaited_once()
    assert isinstance(result, dict)


@pytest.mark.asyncio
async def test_execute_model_completion_raises_on_error(mock_user_key):
    svc = PromptCompletionService(mock_user_key)
    with patch("litellm.acompletion", side_effect=Exception("boom")):
        with pytest.raises(HTTPException) as e:
            await svc.execute_model_completion("m", ("m", []), {})
        assert e.value.status_code == 500


# -------------------------------------------------------------------
# Full orchestration (happy path)
# -------------------------------------------------------------------
@patch("litellm.proxy.prompts.prompt_completion_service.PROMPT_HUB")
@patch("litellm.acompletion", new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_run_completion_happy(mock_acompletion, mock_hub, mock_user_key):
    mock_prompt_spec = MagicMock()
    mock_prompt_spec.model_dump.return_value = {"prompt_id": "promptA"}
    mock_hub.get_prompt_by_id.return_value = mock_prompt_spec

    mock_callback = MagicMock(spec=DotpromptManager)
    mock_callback.prompt_manager.get_all_prompts_as_json.return_value = {
        "promptA": {"content": "hi", "metadata": {"model": "gpt-4"}}
    }
    mock_hub.get_prompt_callback_by_id.return_value = mock_callback

    svc = PromptCompletionService(mock_user_key)
    svc.merge_params = MagicMock(return_value={"user": "u1"})

    mock_acompletion.return_value = {
        "choices": [{"message": {"content": "Hello world"}}]
    }

    result = await svc.run_completion("promptA", {"x": 1}, "v1", {"foo": "bar"})
    assert result.completion_text == "Hello world"

