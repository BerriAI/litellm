from litellm.litellm_core_utils.served_output_texts import (
    SERVED_OUTPUT_TEXTS_KEY,
    overlay_served_output_texts,
    record_served_output_texts,
    served_output_texts,
    served_stream_output_texts,
)
from litellm.types.utils import Choices, Delta, Message, ModelResponse, ModelResponseStream, StreamingChoices

RAW = "Card: 4111 1111 1111 1111"
MASKED = "Card: <CREDIT_CARD>"


def _chat_response(*texts: str) -> ModelResponse:
    return ModelResponse(
        choices=[Choices(index=i, message=Message(content=text, role="assistant")) for i, text in enumerate(texts)]
    )


def _chat_dict(*texts: str) -> dict[str, object]:
    return {
        "id": "x",
        "choices": [
            {"index": i, "finish_reason": "stop", "message": {"role": "assistant", "content": text}}
            for i, text in enumerate(texts)
        ],
    }


def _choice_texts(response: object) -> tuple[str | None, ...]:
    texts = served_output_texts(response)
    assert texts is not None, response
    return texts


def _stream_chunk(text: str, index: int = 0) -> ModelResponseStream:
    return ModelResponseStream(choices=[StreamingChoices(index=index, delta=Delta(content=text))])


def test_served_output_texts_reads_each_response_shape():
    assert served_output_texts(_chat_response(MASKED, "second")) == (MASKED, "second")
    assert served_output_texts(_chat_dict(MASKED)) == (MASKED,)
    assert served_output_texts(
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}],
        }
    ) == ("ab",)
    responses_api: dict[str, object] = {
        "object": "response",
        "output": [
            {"type": "reasoning", "content": []},
            {"type": "message", "content": [{"type": "output_text", "text": MASKED}]},
        ],
    }
    assert served_output_texts(responses_api) == (MASKED,)
    assert served_output_texts({"data": [{"embedding": [0.1]}]}) is None
    assert served_output_texts("plain") is None


def test_served_stream_output_texts_joins_chat_chunks_and_reads_anthropic_sse():
    assert served_stream_output_texts([_stream_chunk("Card: "), _stream_chunk("<CREDIT_CARD>")]) == (MASKED,)
    sse = (
        'event: message_start\ndata: {"type":"message_start","message":{"id":"m","type":"message","role":"assistant",'
        '"content":[],"model":"x","usage":{"input_tokens":1,"output_tokens":1}}}\n\n',
        'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
        f'event: content_block_delta\ndata: {{"type":"content_block_delta","index":0,"delta":{{"type":"text_delta","text":"{MASKED}"}}}}\n\n',
        'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
        'event: message_stop\ndata: {"type":"message_stop"}\n\n',
    )
    assert served_stream_output_texts(tuple(chunk.encode() for chunk in sse)) == (MASKED,)
    assert served_stream_output_texts([]) is None
    assert served_stream_output_texts([b"not sse"]) is None


def test_served_stream_output_texts_keeps_every_choice_index_when_chunks_carry_one_choice_each():
    chunks = [_stream_chunk("first ", 0), _stream_chunk(MASKED, 1), _stream_chunk("choice", 0)]
    assert served_stream_output_texts(chunks) == ("first choice", MASKED)


def test_blanked_output_is_served_as_empty_text_and_overlaid():
    assert served_output_texts(_chat_response("")) == ("",)
    assert served_output_texts({"type": "message", "role": "assistant", "content": [{"type": "text", "text": ""}]}) == (
        "",
    )
    assert served_stream_output_texts([_stream_chunk("")]) == ("",)
    assert _choice_texts(overlay_served_output_texts(_chat_dict(RAW), ("",))) == ("",)


def test_overlay_replaces_logged_choice_text_with_served_text():
    logged = _chat_dict(RAW, RAW)
    overlaid = overlay_served_output_texts(logged, (MASKED,))
    assert _choice_texts(overlaid) == (MASKED, RAW)
    assert isinstance(overlaid, dict)
    assert overlaid["id"] == "x"
    assert _choice_texts(logged) == (RAW, RAW)


def test_overlay_leaves_unreadable_inputs_untouched():
    logged = _chat_dict(RAW)
    assert overlay_served_output_texts(logged, None) is logged
    assert overlay_served_output_texts(logged, "not a tuple") is logged
    assert overlay_served_output_texts(logged, (None,)) == logged
    assert overlay_served_output_texts("text", (MASKED,)) == "text"
    assert overlay_served_output_texts({"data": []}, (MASKED,)) == {"data": []}


def test_record_served_output_texts_only_writes_readable_texts():
    details: dict[str, object] = {}
    record_served_output_texts(details, None)
    assert SERVED_OUTPUT_TEXTS_KEY not in details
    record_served_output_texts(details, (MASKED,))
    assert details[SERVED_OUTPUT_TEXTS_KEY] == (MASKED,)
