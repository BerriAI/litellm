import os
import sys
import pytest


sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from unittest.mock import patch, MagicMock


import pytest
from unittest.mock import patch, MagicMock
import sys


@pytest.fixture
def fake_prompt_dicts():
    return [
        {
            "prompt_id": "p1",
            "litellm_params": {
                "prompt_id": "p1",
                "prompt_integration": "in_memory",
                "model_config": {"model": "gpt-4"},
            },
            "prompt_info": {"prompt_type": "config"},
        },
        {
            "prompt_id": "p2",
            "litellm_params": {
                "prompt_id": "p2",
                "prompt_integration": "in_memory",
                "model_config": {"model": "gpt-3.5"},
            },
            "prompt_info": {"prompt_type": "config"},
        },
    ]


def test_init_prompts_calls_registry_for_each(fake_prompt_dicts):
    """Ensure it calls initialize_prompt() once per prompt."""
    from litellm.proxy.prompts import init_prompts as mod

    fake_reg = MagicMock()
    fake_spec_cls = MagicMock()
    fake_spec_instance = MagicMock()
    fake_spec_cls.side_effect = [fake_spec_instance, fake_spec_instance]
    fake_reg.initialize_prompt.side_effect = [fake_spec_instance, fake_spec_instance]

    with patch.dict(sys.modules, {
        "litellm.types.prompts.init_prompts": MagicMock(PromptSpec=fake_spec_cls),
        "litellm.proxy.prompts.prompt_registry": MagicMock(IN_MEMORY_PROMPT_REGISTRY=fake_reg),
    }):
        result = mod.init_prompts(fake_prompt_dicts, config_file_path="/tmp/config.yml")

    assert result == [fake_spec_instance, fake_spec_instance]
    assert fake_reg.initialize_prompt.call_count == 2
    for call in fake_reg.initialize_prompt.call_args_list:
        assert "config_file_path" in call.kwargs
        assert call.kwargs["config_file_path"] == "/tmp/config.yml"


def test_init_prompts_skips_none_results(fake_prompt_dicts):
    """Should skip prompts returning None."""
    from litellm.proxy.prompts import init_prompts as mod

    fake_reg = MagicMock()
    fake_spec_cls = MagicMock()
    fake_spec_instance = MagicMock()
    fake_reg.initialize_prompt.side_effect = [None, fake_spec_instance]

    with patch.dict(sys.modules, {
        "litellm.types.prompts.init_prompts": MagicMock(PromptSpec=fake_spec_cls),
        "litellm.proxy.prompts.prompt_registry": MagicMock(IN_MEMORY_PROMPT_REGISTRY=fake_reg),
    }):
        result = mod.init_prompts(fake_prompt_dicts)

    assert len(result) == 1
    assert result[0] is fake_spec_instance


def test_init_prompts_empty_list_returns_empty():
    """If no prompts are provided, returns [] and doesn’t call registry."""
    from litellm.proxy.prompts import init_prompts as mod
    fake_reg = MagicMock()

    with patch.dict(sys.modules, {
        "litellm.types.prompts.init_prompts": MagicMock(PromptSpec=MagicMock()),
        "litellm.proxy.prompts.prompt_registry": MagicMock(IN_MEMORY_PROMPT_REGISTRY=fake_reg),
    }):
        result = mod.init_prompts([])

    fake_reg.initialize_prompt.assert_not_called()
    assert result == []


def test_init_prompts_creates_prompt_spec_instances(fake_prompt_dicts):
    """Verify PromptSpec(**dict) construction and delegation."""
    from litellm.proxy.prompts import init_prompts as mod

    fake_spec_cls = MagicMock()
    fake_spec_instance = MagicMock()
    fake_spec_cls.return_value = fake_spec_instance
    fake_reg = MagicMock()
    fake_reg.initialize_prompt.return_value = fake_spec_instance

    with patch.dict(sys.modules, {
        "litellm.types.prompts.init_prompts": MagicMock(PromptSpec=fake_spec_cls),
        "litellm.proxy.prompts.prompt_registry": MagicMock(IN_MEMORY_PROMPT_REGISTRY=fake_reg),
    }):
        result = mod.init_prompts(fake_prompt_dicts)

    assert all(r is fake_spec_instance for r in result)
    assert fake_spec_cls.call_count == len(fake_prompt_dicts)
