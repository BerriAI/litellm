from collections.abc import Coroutine, Mapping, Sequence
from typing import Literal, overload

import httpx

from litellm.types.tokenize import TokenizeRepresentation, TokenizeResponse

__all__ = ("atokenize", "tokenize")


@overload
def tokenize(
    model: str,
    prompt: str,
    api_key: str | None = ...,
    api_base: str | None = ...,
    timeout: float | httpx.Timeout | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: Mapping[str, object] | None = ...,
    include: Sequence[TokenizeRepresentation] = ...,
    *,
    add_special_tokens: bool = ...,
    atokenize: Literal[False] = ...,
    metadata: Mapping[str, object] | None = ...,
    num_retries: int | None = ...,
    user: str | None = ...,
    **kwargs: object,
) -> TokenizeResponse: ...


@overload
def tokenize(
    model: str,
    prompt: str,
    api_key: str | None = ...,
    api_base: str | None = ...,
    timeout: float | httpx.Timeout | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: Mapping[str, object] | None = ...,
    include: Sequence[TokenizeRepresentation] = ...,
    *,
    add_special_tokens: bool = ...,
    atokenize: Literal[True],
    metadata: Mapping[str, object] | None = ...,
    num_retries: int | None = ...,
    user: str | None = ...,
    **kwargs: object,
) -> Coroutine[object, object, TokenizeResponse]: ...


@overload
def tokenize(
    model: str,
    prompt: str,
    api_key: str | None = ...,
    api_base: str | None = ...,
    timeout: float | httpx.Timeout | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: Mapping[str, object] | None = ...,
    include: Sequence[TokenizeRepresentation] = ...,
    *,
    add_special_tokens: bool = ...,
    atokenize: bool,
    metadata: Mapping[str, object] | None = ...,
    num_retries: int | None = ...,
    user: str | None = ...,
    **kwargs: object,
) -> TokenizeResponse | Coroutine[object, object, TokenizeResponse]: ...


async def atokenize(
    model: str,
    prompt: str,
    api_key: str | None = ...,
    api_base: str | None = ...,
    timeout: float | httpx.Timeout | None = ...,
    custom_llm_provider: str | None = ...,
    extra_headers: Mapping[str, object] | None = ...,
    include: Sequence[TokenizeRepresentation] = ...,
    *,
    add_special_tokens: bool = ...,
    atokenize: bool = ...,
    metadata: Mapping[str, object] | None = ...,
    num_retries: int | None = ...,
    user: str | None = ...,
    **kwargs: object,
) -> TokenizeResponse: ...
