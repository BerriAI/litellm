"""
Test OpenAI Embeddings Guardrail Translation Handler
"""

import copy
from collections.abc import Sequence
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm.integrations.custom_guardrail import CustomGuardrail
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.guardrail_translation.utils import UnappliableRequestRewrite
from litellm.llms.openai.embeddings.guardrail_translation.handler import (
    OpenAIEmbeddingsHandler,
)
from litellm.types.utils import CallTypes, GenericGuardrailAPIInputs


@pytest.mark.asyncio
async def test_embeddings_handler_string_input():
    """Test embeddings handler with single string input"""
    handler = OpenAIEmbeddingsHandler()

    # Mock guardrail
    mock_guardrail = MagicMock()
    mock_guardrail.apply_guardrail = AsyncMock(
        return_value={"texts": ["processed text"]}
    )

    data = {"input": "Hello, world!", "model": "text-embedding-3-small"}

    result = await handler.process_input_messages(
        data=data,
        guardrail_to_apply=mock_guardrail,
    )

    # Verify guardrail was called with correct inputs
    mock_guardrail.apply_guardrail.assert_called_once()
    call_args = mock_guardrail.apply_guardrail.call_args
    assert call_args.kwargs["inputs"]["texts"] == ["Hello, world!"]
    assert call_args.kwargs["inputs"]["model"] == "text-embedding-3-small"

    # Verify result
    assert result["input"] == "processed text"


@pytest.mark.asyncio
async def test_embeddings_handler_list_of_strings_input():
    """Test embeddings handler with list of strings input"""
    handler = OpenAIEmbeddingsHandler()

    # Mock guardrail
    mock_guardrail = MagicMock()
    mock_guardrail.apply_guardrail = AsyncMock(
        return_value={"texts": ["processed text 1", "processed text 2"]}
    )

    data = {
        "input": ["Hello, world!", "How are you?"],
        "model": "text-embedding-3-small",
    }

    result = await handler.process_input_messages(
        data=data,
        guardrail_to_apply=mock_guardrail,
    )

    # Verify guardrail was called with correct inputs
    mock_guardrail.apply_guardrail.assert_called_once()
    call_args = mock_guardrail.apply_guardrail.call_args
    assert call_args.kwargs["inputs"]["texts"] == ["Hello, world!", "How are you?"]

    # Verify result
    assert result["input"] == ["processed text 1", "processed text 2"]


def test_embeddings_guardrail_translation_mappings():
    """Test that embeddings handler is registered for correct call types"""
    from litellm.llms.openai.embeddings.guardrail_translation import (
        guardrail_translation_mappings,
    )

    assert CallTypes.embedding in guardrail_translation_mappings
    assert CallTypes.aembedding in guardrail_translation_mappings
    assert (
        guardrail_translation_mappings[CallTypes.embedding] == OpenAIEmbeddingsHandler
    )
    assert (
        guardrail_translation_mappings[CallTypes.aembedding] == OpenAIEmbeddingsHandler
    )


SECRET: Final = "sk-live-0123456789"
MASK: Final = "[REDACTED]"
FILE_BLOCK: Final = {
    "type": "file",
    "file": {"file_data": "data:video/mp4;base64,AAAA", "video_metadata": {"fps": 1}},
}


class _MaskingGuardrail(CustomGuardrail):
    """Masks SECRET in every text it is handed and remembers what it saw."""

    def __init__(self) -> None:
        super().__init__(guardrail_name="masking-test-guardrail")
        self.seen: tuple[str, ...] | None = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        self.seen = tuple(inputs.get("texts", []))
        return GenericGuardrailAPIInputs(texts=[text.replace(SECRET, MASK) for text in self.seen])


class _BlockedError(Exception):
    pass


class _BlockingGuardrail(CustomGuardrail):
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        raise _BlockedError("secret found")


class _RecordingFixedTextsGuardrail(CustomGuardrail):
    """Returns fixed texts and keeps the raw `texts` value it was handed."""

    def __init__(self, texts: Sequence[str]) -> None:
        super().__init__(guardrail_name="recording-test-guardrail")
        self.texts: Final = texts
        self.raw_texts: object = None

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        self.raw_texts = inputs.get("texts")
        return GenericGuardrailAPIInputs(texts=list(self.texts))


class _FixedTextsGuardrail(CustomGuardrail):
    def __init__(self, texts: Sequence[str]) -> None:
        super().__init__(guardrail_name="fixed-texts-test-guardrail")
        self.texts: Final = texts

    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        return GenericGuardrailAPIInputs(texts=list(self.texts))


def _text_block(text: str) -> dict[str, object]:
    return {"type": "text", "text": text}


@pytest.mark.parametrize(
    "embedding_input, expected_seen, expected_input",
    [
        pytest.param(
            [{"type": "text", "text": f"key {SECRET}", "cache_control": {"type": "ephemeral"}}, _text_block("plain")],
            (f"key {SECRET}", "plain"),
            [{"type": "text", "text": f"key {MASK}", "cache_control": {"type": "ephemeral"}}, _text_block("plain")],
            id="text-block-first",
        ),
        pytest.param(
            [f"a {SECRET}", _text_block(f"b {SECRET}"), FILE_BLOCK],
            (f"a {SECRET}", f"b {SECRET}"),
            [f"a {MASK}", _text_block(f"b {MASK}"), FILE_BLOCK],
            id="string-first-mixed",
        ),
        pytest.param(
            [[_text_block(SECRET), FILE_BLOCK], f"x {SECRET}"],
            (SECRET, f"x {SECRET}"),
            [[_text_block(MASK), FILE_BLOCK], f"x {MASK}"],
            id="grouped",
        ),
        pytest.param([_text_block("")], ("",), [_text_block("")], id="empty-text"),
        pytest.param(
            [f"a {SECRET}", _text_block(f"b {SECRET}"), {"type": "text", "text": 5}],
            (f"a {SECRET}", f"b {SECRET}"),
            [f"a {MASK}", _text_block(f"b {MASK}"), {"type": "text", "text": 5}],
            id="malformed-text-block-beside-a-valid-one",
        ),
    ],
)
@pytest.mark.asyncio
async def test_text_blocks_are_guardrailed_and_rewritten_in_place(
    embedding_input: Sequence[object], expected_seen: tuple[str, ...], expected_input: Sequence[object]
) -> None:
    guardrail: Final = _MaskingGuardrail()
    data: Final = {"input": embedding_input, "model": "gemini/gemini-embedding-2"}

    result: Final = await OpenAIEmbeddingsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert guardrail.seen == expected_seen
    assert data["input"] == expected_input
    assert result["input"] == expected_input


@pytest.mark.asyncio
async def test_a_guardrail_error_on_text_blocks_propagates_and_leaves_the_input_alone() -> None:
    embedding_input: Final = [_text_block(SECRET), FILE_BLOCK]
    data: Final = {"input": copy.deepcopy(embedding_input)}

    with pytest.raises(_BlockedError):
        await OpenAIEmbeddingsHandler().process_input_messages(data=data, guardrail_to_apply=_BlockingGuardrail())

    assert data["input"] == embedding_input


@pytest.mark.asyncio
async def test_a_rewrite_with_the_wrong_number_of_texts_fails_closed() -> None:
    embedding_input: Final = [_text_block(SECRET), f"x {SECRET}"]
    data: Final = {"input": copy.deepcopy(embedding_input)}

    with pytest.raises(UnappliableRequestRewrite, match="fixed-texts-test-guardrail"):
        await OpenAIEmbeddingsHandler().process_input_messages(
            data=data, guardrail_to_apply=_FixedTextsGuardrail([MASK])
        )

    assert data["input"] == embedding_input


@pytest.mark.asyncio
async def test_a_rewrite_with_no_texts_leaves_text_blocks_unchanged() -> None:
    embedding_input: Final = [_text_block("hello"), "world"]
    data: Final = {"input": copy.deepcopy(embedding_input)}

    await OpenAIEmbeddingsHandler().process_input_messages(data=data, guardrail_to_apply=_FixedTextsGuardrail([]))

    assert data["input"] == embedding_input


@pytest.mark.parametrize(
    "embedding_input",
    [
        pytest.param([1, 2, 3], id="token-ids"),
        pytest.param([[1, 2], [3]], id="nested-token-ids"),
        pytest.param([FILE_BLOCK], id="file-block-only"),
        pytest.param([{"type": "text", "text": 5}], id="malformed-text-block-only"),
    ],
)
@pytest.mark.asyncio
async def test_inputs_without_text_blocks_keep_skipping_the_guardrail(embedding_input: Sequence[object]) -> None:
    guardrail: Final = _MaskingGuardrail()
    data: Final = {"input": embedding_input}

    await OpenAIEmbeddingsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert guardrail.seen is None
    assert data["input"] is embedding_input


@pytest.mark.parametrize(
    "embedding_input",
    [
        pytest.param([f"a {SECRET}", FILE_BLOCK], id="string-then-file-block"),
        pytest.param([f"a {SECRET}", {"type": "text", "text": 5}], id="string-then-malformed-text-block"),
        pytest.param([f"a {SECRET}", 7], id="string-then-token-id"),
    ],
)
@pytest.mark.asyncio
async def test_string_first_lists_without_text_blocks_keep_the_raw_guardrail_contract(
    embedding_input: Sequence[object],
) -> None:
    guardrail: Final = _RecordingFixedTextsGuardrail(["guardrailed"])
    data: Final = {"input": embedding_input}

    await OpenAIEmbeddingsHandler().process_input_messages(data=data, guardrail_to_apply=guardrail)

    assert guardrail.raw_texts is embedding_input
    assert data["input"] == ["guardrailed"]


@pytest.mark.asyncio
async def test_a_plain_string_list_still_takes_the_guardrail_texts_whatever_their_count() -> None:
    data: Final = {"input": ["a", "b"]}

    await OpenAIEmbeddingsHandler().process_input_messages(
        data=data, guardrail_to_apply=_FixedTextsGuardrail(["a and b"])
    )

    assert data["input"] == ["a and b"]


@pytest.mark.parametrize(
    "embedding_input, expected_input",
    [
        pytest.param(f"a {SECRET}", f"a {MASK}", id="string"),
        pytest.param([f"a {SECRET}", "b"], [f"a {MASK}", "b"], id="list-of-strings"),
    ],
)
@pytest.mark.asyncio
async def test_plain_string_inputs_are_still_masked(
    embedding_input: str | Sequence[str], expected_input: object
) -> None:
    data: Final = {"input": embedding_input}

    await OpenAIEmbeddingsHandler().process_input_messages(data=data, guardrail_to_apply=_MaskingGuardrail())

    assert data["input"] == expected_input
