from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from typing import Final

import httpx

from litellm.rust_bridge import runtime
from litellm.rust_bridge.batches.native import (
    NATIVE_ACREATE_BATCH,
    NATIVE_ARETRIEVE_BATCH,
    NATIVE_CREATE_BATCH,
    NATIVE_RETRIEVE_BATCH,
    RustAcreateBatch,
    RustAretrieveBatch,
    RustCreateBatch,
    RustRetrieveBatch,
)
from litellm.rust_bridge.catalog import Route, RouteContext
from litellm.rust_bridge.timeouts import timeout_to_seconds
from litellm.types.utils import LiteLLMBatch

ANTHROPIC_CONTEXT: Final = RouteContext(Route.BATCHES, provider="anthropic")
ANTHROPIC_BATCH_INPUT_CONTENT_KWARG: Final = "litellm_batch_input_content"

BatchResult = LiteLLMBatch | Coroutine[object, object, LiteLLMBatch]


@dataclass(frozen=True, slots=True)
class AnthropicConnection:
    api_key: str | None
    api_base: str | None
    extra_headers: dict[str, str] | None
    timeout: float | httpx.Timeout | None


def retrieve_anthropic_batch(
    *,
    is_async: bool,
    batch_id: str,
    connection: AnthropicConnection,
    python: Callable[[], BatchResult],
) -> BatchResult:
    timeout_seconds: Final = timeout_to_seconds(connection.timeout)

    def native(rust: RustRetrieveBatch) -> LiteLLMBatch:
        return LiteLLMBatch.model_validate(
            rust(batch_id, connection.api_key, connection.api_base, connection.extra_headers, timeout_seconds)
        )

    async def anative(rust: RustAretrieveBatch) -> LiteLLMBatch:
        return LiteLLMBatch.model_validate(
            await rust(batch_id, connection.api_key, connection.api_base, connection.extra_headers, timeout_seconds)
        )

    async def apython() -> LiteLLMBatch:
        result: Final = python()
        return await result if isinstance(result, Coroutine) else result

    if is_async:
        return runtime.arun(ANTHROPIC_CONTEXT, binding=NATIVE_ARETRIEVE_BATCH, native=anative, python=apython)
    return runtime.run(ANTHROPIC_CONTEXT, binding=NATIVE_RETRIEVE_BATCH, native=native, python=python)


def create_anthropic_batch(
    *,
    is_async: bool,
    input_jsonl: str | None,
    model: str | None,
    connection: AnthropicConnection,
    python: Callable[[], LiteLLMBatch],
) -> BatchResult:
    timeout_seconds: Final = timeout_to_seconds(connection.timeout)

    def content() -> str:
        if input_jsonl is None:
            raise ValueError(
                "Anthropic batches read their requests from a managed file: upload the input file through the "
                "proxy with purpose=batch, target_model_names, and a target_storage backend"
            )
        return input_jsonl

    def native(rust: RustCreateBatch) -> LiteLLMBatch:
        return LiteLLMBatch.model_validate(
            rust(
                content(),
                model,
                connection.api_key,
                connection.api_base,
                connection.extra_headers,
                timeout_seconds,
            )
        )

    async def anative(rust: RustAcreateBatch) -> LiteLLMBatch:
        return LiteLLMBatch.model_validate(
            await rust(
                content(),
                model,
                connection.api_key,
                connection.api_base,
                connection.extra_headers,
                timeout_seconds,
            )
        )

    async def apython() -> LiteLLMBatch:
        return python()

    if is_async:
        return runtime.arun(ANTHROPIC_CONTEXT, binding=NATIVE_ACREATE_BATCH, native=anative, python=apython)
    return runtime.run(ANTHROPIC_CONTEXT, binding=NATIVE_CREATE_BATCH, native=native, python=python)
