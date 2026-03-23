"""
Tests for handling malformed 'file' content blocks (missing 'file' sub-field).

Regression tests for:
- litellm/llms/vertex_ai/gemini/transformation.py
- litellm/llms/gemini/chat/transformation.py
- litellm/litellm_core_utils/prompt_templates/common_utils.py
- litellm/litellm_core_utils/prompt_templates/factory.py (Bedrock + Anthropic)
- litellm/llms/openai/chat/gpt_transformation.py
"""

import copy
from typing import List, cast

import pytest

import litellm
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_file_ids_from_messages,
    migrate_file_to_image_url,
    update_messages_with_model_file_ids,
)
from litellm.litellm_core_utils.prompt_templates.factory import (
    BedrockConverseMessagesProcessor,
    anthropic_process_openai_file_message,
)
from litellm.llms.gemini.chat.transformation import GoogleAIStudioGeminiConfig
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,
)
from litellm.types.llms.openai import (
    AllMessageValues,
    ChatCompletionFileObject,
    OpenAIMessageContentListBlock,
)

_MALFORMED_MESSAGES_RAW = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "hello"},
            {"type": "file"},  # Missing required "file" sub-field
        ],
    }
]

_WELL_FORMED_MESSAGES_RAW = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "hello"},
            {
                "type": "file",
                "file": {"file_id": "file-abc123", "format": "pdf"},
            },
        ],
    }
]

MALFORMED_FILE_OBJECT: ChatCompletionFileObject = cast(
    ChatCompletionFileObject, {"type": "file"}
)


def _malformed() -> List[AllMessageValues]:
    return copy.deepcopy(cast(List[AllMessageValues], _MALFORMED_MESSAGES_RAW))


def _well_formed() -> List[AllMessageValues]:
    return copy.deepcopy(cast(List[AllMessageValues], _WELL_FORMED_MESSAGES_RAW))


# ---------------------------------------------------------------------------
# vertex_ai/gemini/transformation.py
# ---------------------------------------------------------------------------


def test_gemini_convert_messages_malformed_file_raises_bad_request():
    """_gemini_convert_messages_with_history should raise BadRequestError (not KeyError)
    when a content block has type='file' but no 'file' sub-field."""
    with pytest.raises(litellm.BadRequestError, match="missing the required 'file' field"):
        _gemini_convert_messages_with_history(
            messages=_malformed(),
            model="gemini-2.0-flash",
        )


# ---------------------------------------------------------------------------
# gemini/chat/transformation.py - GoogleAIStudioGeminiConfig
# ---------------------------------------------------------------------------


def test_google_ai_studio_transform_messages_malformed_file_raises_bad_request():
    """GoogleAIStudioGeminiConfig._transform_messages should raise BadRequestError
    when a content block has type='file' but no 'file' sub-field."""
    config = GoogleAIStudioGeminiConfig()
    with pytest.raises(litellm.BadRequestError, match="missing the required 'file' field"):
        config._transform_messages(messages=_malformed(), model="gemini-2.0-flash")


# ---------------------------------------------------------------------------
# common_utils.py - update_messages_with_model_file_ids
# ---------------------------------------------------------------------------


def test_update_messages_with_model_file_ids_malformed_raises_bad_request():
    """update_messages_with_model_file_ids should raise BadRequestError for content
    blocks that have type='file' but no 'file' sub-field."""
    with pytest.raises(litellm.BadRequestError, match="missing the required 'file' field"):
        update_messages_with_model_file_ids(
            messages=_malformed(),
            model_id="some-model",
            model_file_id_mapping={},
        )


def test_update_messages_with_model_file_ids_well_formed_updates():
    """update_messages_with_model_file_ids should update file_id for well-formed blocks."""
    mapping = {"file-abc123": {"some-model": "provider-file-xyz"}}
    result = update_messages_with_model_file_ids(
        messages=_well_formed(),
        model_id="some-model",
        model_file_id_mapping=mapping,
    )
    content = result[0].get("content")
    assert isinstance(content, list)
    file_block = next(c for c in content if c.get("type") == "file")
    assert file_block.get("file", {}).get("file_id") == "provider-file-xyz"


# ---------------------------------------------------------------------------
# common_utils.py - get_file_ids_from_messages
# ---------------------------------------------------------------------------


def test_get_file_ids_from_messages_malformed_raises_bad_request():
    """get_file_ids_from_messages should raise BadRequestError for malformed file blocks."""
    with pytest.raises(litellm.BadRequestError, match="missing the required 'file' field"):
        get_file_ids_from_messages(messages=_malformed())


def test_get_file_ids_from_messages_well_formed_returns_ids():
    """get_file_ids_from_messages should extract file_id from well-formed blocks."""
    messages: List[AllMessageValues] = cast(
        List[AllMessageValues],
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "hello"},
                    {"type": "file", "file": {"file_id": "file-abc123", "format": "pdf"}},
                ],
            }
        ],
    )
    result = get_file_ids_from_messages(messages=messages)
    assert result == ["file-abc123"]


# ---------------------------------------------------------------------------
# factory.py - BedrockConverseMessagesProcessor (sync + async)
# ---------------------------------------------------------------------------


def test_bedrock_process_file_message_malformed_raises_bad_request():
    """_process_file_message should raise BadRequestError (not KeyError)
    when the file object is missing the 'file' sub-field."""
    with pytest.raises(litellm.BadRequestError, match="missing the required 'file' field"):
        BedrockConverseMessagesProcessor._process_file_message(MALFORMED_FILE_OBJECT)


@pytest.mark.asyncio
async def test_bedrock_async_process_file_message_malformed_raises_bad_request():
    """_async_process_file_message should raise BadRequestError (not KeyError)
    when the file object is missing the 'file' sub-field."""
    with pytest.raises(litellm.BadRequestError, match="missing the required 'file' field"):
        await BedrockConverseMessagesProcessor._async_process_file_message(
            MALFORMED_FILE_OBJECT
        )


# ---------------------------------------------------------------------------
# openai/chat/gpt_transformation.py
# ---------------------------------------------------------------------------


def test_openai_apply_common_transform_malformed_file_raises_bad_request():
    """_apply_common_transform_content_item should raise BadRequestError (not KeyError)
    when a content block has type='file' but no 'file' sub-field."""
    config = OpenAIGPTConfig()
    malformed_block: OpenAIMessageContentListBlock = cast(
        OpenAIMessageContentListBlock, {"type": "file"}
    )
    with pytest.raises(litellm.BadRequestError, match="missing the required 'file' field"):
        config._apply_common_transform_content_item(malformed_block)


def test_openai_apply_common_transform_well_formed_file_does_not_raise():
    """_apply_common_transform_content_item should not raise for well-formed file blocks."""
    config = OpenAIGPTConfig()
    well_formed_block: OpenAIMessageContentListBlock = cast(
        OpenAIMessageContentListBlock,
        {"type": "file", "file": {"file_id": "file-abc123"}},
    )
    result = config._apply_common_transform_content_item(well_formed_block)
    assert result.get("type") == "file"
    file_field = cast(ChatCompletionFileObject, result).get("file", {})
    assert file_field.get("file_id") == "file-abc123"


# ---------------------------------------------------------------------------
# factory.py - anthropic_process_openai_file_message
# ---------------------------------------------------------------------------


def test_anthropic_process_openai_file_message_malformed_raises_bad_request():
    """anthropic_process_openai_file_message should raise BadRequestError (not KeyError)
    when the file object is missing the 'file' sub-field."""
    with pytest.raises(litellm.BadRequestError, match="missing the required 'file' field"):
        anthropic_process_openai_file_message(MALFORMED_FILE_OBJECT)


def test_anthropic_process_openai_file_message_well_formed_file_id_does_not_raise():
    """anthropic_process_openai_file_message should not raise for a well-formed file_id block."""
    well_formed: ChatCompletionFileObject = cast(
        ChatCompletionFileObject,
        {"type": "file", "file": {"file_id": "file-abc123"}},
    )
    result = anthropic_process_openai_file_message(well_formed)
    assert result.get("type") in ("document", "image", "container_upload")


# ---------------------------------------------------------------------------
# common_utils.py - migrate_file_to_image_url
# ---------------------------------------------------------------------------


def test_migrate_file_to_image_url_malformed_raises_bad_request():
    """migrate_file_to_image_url should raise BadRequestError (not KeyError)
    when the file object is missing the 'file' sub-field."""
    with pytest.raises(litellm.BadRequestError, match="missing the required 'file' field"):
        migrate_file_to_image_url(MALFORMED_FILE_OBJECT)


def test_migrate_file_to_image_url_well_formed_returns_image_url():
    """migrate_file_to_image_url should return an image_url block for a well-formed file."""
    well_formed: ChatCompletionFileObject = cast(
        ChatCompletionFileObject,
        {"type": "file", "file": {"file_id": "file-abc123", "format": "png"}},
    )
    result = migrate_file_to_image_url(well_formed)
    assert result.get("type") == "image_url"
    image_url = result.get("image_url", {})
    assert isinstance(image_url, dict)
    assert image_url.get("url") == "file-abc123"
