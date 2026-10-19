from typing import Final

import pytest

from litellm.litellm_core_utils.prompt_templates.chat_template_tools import chat_template_tool_section

_DESCRIBED_FUNCTION: Final = {"name": "f", "description": "é", "parameters": {}}
_BARE_FUNCTION: Final = {"name": "g", "parameters": {}}


# Tool schemas as rendered by render_tools in encoding/encoding_dsv4.py (deepseek-ai/DeepSeek-V4-Flash) and by
# chat_template.jinja (Qwen/Qwen3.5-122B-A10B), both read on Hugging Face on 2026-10-17
@pytest.mark.parametrize(
    ("model", "rendered"),
    [
        (
            "DeepSeek-V4-Flash-0731",
            '{"name": "f", "description": "é", "parameters": {}}\n{"name": "g", "parameters": {}}',
        ),
        (
            "databricks-qwen35-122b-a10b",
            '\n{"type": "function", "function": {"name": "f", "description": "é", "parameters": {}}}'
            '\n{"type": "function", "function": {"name": "g", "parameters": {}}}',
        ),
    ],
)
def test_chat_template_tool_section_renders_schemas_like_the_models_template(model: str, rendered: str):
    section: Final = chat_template_tool_section(model)

    assert section is not None
    assert section.render_tool_schemas((_DESCRIBED_FUNCTION, _BARE_FUNCTION)) == rendered
