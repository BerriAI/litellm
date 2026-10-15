from collections.abc import Coroutine

from typing_extensions import assert_type

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.types.utils import EmbeddingResponse


def embedding_dispatch_types() -> None:
    assert_type(
        litellm.embedding(model="m", input=["x"]),
        EmbeddingResponse,
    )
    pending = litellm.embedding(model="m", input=["x"], aembedding=True)
    assert_type(pending, Coroutine[object, object, EmbeddingResponse])  # pyright: ignore[reportUnusedCoroutine]  # the coroutine is produced only to assert its type


def anthropic_messages_dispatch_rejects_wrong_types() -> None:
    litellm.anthropic_messages_handler(max_tokens="x", messages=[], model="m")  # pyright: ignore[reportArgumentType, reportUnknownMemberType]  # max_tokens is typed int; legacy signature leaves other params unannotated


async def responses_dispatch_types() -> None:
    from litellm.responses.streaming_iterator import BaseResponsesAPIStreamingIterator
    from litellm.types.llms.openai import ResponsesAPIResponse

    assert_type(
        await litellm.aresponses(input="x", model="m"),  # pyright: ignore[reportUnknownMemberType]  # legacy signature leaves text and **kwargs unannotated
        ResponsesAPIResponse | BaseResponsesAPIStreamingIterator,
    )
    litellm.responses(input="x", model=1)  # pyright: ignore[reportArgumentType, reportUnknownMemberType]  # model is typed str; legacy signature leaves text and **kwargs unannotated


async def ocr_dispatch_types() -> None:
    document = {"type": "url", "url": "https://example.com/doc.pdf"}
    assert_type(litellm.ocr(model="m", document=document), OCRResponse)
    assert_type(await litellm.aocr(model="m", document=document), OCRResponse)
    litellm.ocr(model=1, document={})  # pyright: ignore[reportArgumentType]  # model is typed str on the OCR binder
