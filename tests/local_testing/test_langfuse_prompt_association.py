import sys
from types import ModuleType
from unittest.mock import MagicMock

from litellm.integrations.langfuse.langfuse import _add_prompt_to_generation_params
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup


def test_standard_logging_metadata_preserves_prompt_selection():
    metadata = StandardLoggingPayloadSetup.get_standard_logging_metadata(
        metadata={},
        litellm_params={
            "prompt_id": "support-agent",
            "prompt_label": "production",
            "prompt_version": 3,
        },
        prompt_integration="langfuse",
    )

    assert metadata["prompt_management_metadata"] == {
        "prompt_id": "support-agent",
        "prompt_variables": None,
        "prompt_integration": "langfuse",
        "prompt_label": "production",
        "prompt_version": 3,
    }


def test_add_prompt_uses_preserved_prompt_selection(monkeypatch):
    langfuse_module = ModuleType("langfuse")
    langfuse_module.Langfuse = MagicMock
    langfuse_model_module = ModuleType("langfuse.model")
    langfuse_model_module.ChatPromptClient = MagicMock
    langfuse_model_module.Prompt_Chat = MagicMock
    langfuse_model_module.Prompt_Text = MagicMock
    langfuse_model_module.TextPromptClient = MagicMock
    monkeypatch.setitem(sys.modules, "langfuse", langfuse_module)
    monkeypatch.setitem(sys.modules, "langfuse.model", langfuse_model_module)

    client = MagicMock()
    prompt = MagicMock()
    client.get_prompt.return_value = prompt

    generation_params = _add_prompt_to_generation_params(
        generation_params={},
        clean_metadata={},
        prompt_management_metadata={
            "prompt_id": "support-agent",
            "prompt_variables": None,
            "prompt_integration": "langfuse",
            "prompt_label": "production",
            "prompt_version": 3,
        },
        langfuse_client=client,
    )

    client.get_prompt.assert_called_once_with("support-agent", label="production", version=3)
    assert generation_params["prompt"] is prompt
