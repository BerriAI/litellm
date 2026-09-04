from __future__ import annotations

from collections.abc import Generator, Mapping
from typing import Final

import pytest

from litellm.rust_bridge import provider_count_tokens
from litellm.types.utils import TokenCountResponse


@pytest.fixture(autouse=True)
def reset_dispatch() -> Generator[None]:
    provider_count_tokens.set_rust_provider_count_tokens(None)
    yield
    provider_count_tokens.set_rust_provider_count_tokens(None)


@pytest.mark.asyncio
async def test_unregistered_route_uses_python_without_native_preparation() -> None:
    events: list[str] = []

    async def fallback() -> TokenCountResponse:
        events.append("python")
        return TokenCountResponse(
            total_tokens=3,
            request_model="model",
            model_used="model",
            tokenizer_type="provider_tokenizer",
        )

    result: Final = await provider_count_tokens.acount_tokens(
        prepare=lambda: events.append("prepare") or {},
        fallback=fallback,
        model="model",
        provider="anthropic",
    )

    assert result is not None
    assert result.total_tokens == 3
    assert events == ["python"]


@pytest.mark.asyncio
async def test_injected_route_owns_one_provider_attempt() -> None:
    events: list[str] = []

    async def native(request: Mapping[str, object]) -> TokenCountResponse:
        events.append(f"native:{request['model']}")
        return TokenCountResponse(
            total_tokens=5,
            request_model="model",
            model_used="model",
            tokenizer_type="provider_tokenizer",
        )

    async def fallback() -> TokenCountResponse:
        pytest.fail("fallback must not run")

    provider_count_tokens.set_rust_provider_count_tokens(native)
    result: Final = await provider_count_tokens.acount_tokens(
        prepare=lambda: events.append("prepare") or {"model": "model"},
        fallback=fallback,
        model="model",
        provider="anthropic",
        request_override=True,
    )

    assert result is not None
    assert result.total_tokens == 5
    assert events == ["prepare", "native:model"]
