"""
Regression tests for partial custom prompt templates.

`initial_prompt_value` and `final_prompt_value` are documented as optional, and
`litellm.completion()` only stores each key when its value is truthy. Call sites that
read them back with direct indexing therefore raised `KeyError` for any template that
set `roles` alone, surfacing as an `APIConnectionError` before a request was ever made.

See https://github.com/BerriAI/litellm/issues/39759
"""

from typing import Final

import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates.factory import response_schema_prompt
from litellm.llms.anthropic.completion.transformation import AnthropicTextConfig
from litellm.llms.ollama.completion.transformation import OllamaConfig
from litellm.llms.predibase.chat.transformation import PredibaseConfig

MODEL: Final = "test-model"

ROLES: Final = {
    "system": {"pre_message": "<<SYS>>\n", "post_message": "\n<</SYS>>\n"},
    "user": {"pre_message": "[INST] ", "post_message": " [/INST]"},
    "assistant": {"pre_message": "", "post_message": "</s>"},
}

MESSAGES: Final = [{"role": "user", "content": "hello"}]

RENDERED_MESSAGES: Final = "[INST] hello [/INST]"


def _ollama_prompt(template: dict) -> str:
    request = OllamaConfig().transform_request(
        model=MODEL,
        messages=MESSAGES,
        optional_params={},
        litellm_params={"custom_prompt_dict": {MODEL: template}},
        headers={},
    )
    return request["prompt"]


@pytest.mark.parametrize(
    "template, expected",
    [
        pytest.param({"roles": ROLES}, RENDERED_MESSAGES, id="roles-only"),
        pytest.param(
            {"roles": ROLES, "initial_prompt_value": "BEGIN\n"},
            f"BEGIN\n{RENDERED_MESSAGES}",
            id="no-final-prompt-value",
        ),
        pytest.param(
            {"roles": ROLES, "final_prompt_value": "\nEND"},
            f"{RENDERED_MESSAGES}\nEND",
            id="no-initial-prompt-value",
        ),
    ],
)
def test_ollama_renders_partial_custom_prompt_template(template: dict, expected: str) -> None:
    """An unset optional key renders as the empty string instead of raising KeyError."""
    assert _ollama_prompt(template) == expected


def test_ollama_full_custom_prompt_template_is_unchanged() -> None:
    """A template that sets every key keeps rendering exactly as before."""
    template = {
        "roles": ROLES,
        "initial_prompt_value": "BEGIN\n",
        "final_prompt_value": "\nEND",
    }

    assert _ollama_prompt(template) == f"BEGIN\n{RENDERED_MESSAGES}\nEND"


def test_predibase_renders_partial_custom_prompt_template() -> None:
    request = PredibaseConfig().transform_request(
        model=MODEL,
        messages=MESSAGES,
        optional_params={},
        litellm_params={"custom_prompt_dict": {MODEL: {"roles": ROLES}}},
        headers={},
    )

    assert request["inputs"] == RENDERED_MESSAGES


def test_anthropic_text_renders_partial_custom_prompt_template(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "custom_prompt_dict", {MODEL: {"roles": ROLES}})

    prompt = AnthropicTextConfig()._get_anthropic_text_prompt_from_messages(
        messages=MESSAGES,
        model=MODEL,
    )

    assert prompt == RENDERED_MESSAGES


def test_response_schema_prompt_renders_partial_custom_prompt_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "custom_prompt_dict", {"response_schema_prompt": {"roles": ROLES}})
    response_schema: Final = {"type": "object", "properties": {"name": {"type": "string"}}}

    prompt = response_schema_prompt(model=MODEL, response_schema=response_schema)

    assert prompt == f"[INST] {response_schema} [/INST]"
