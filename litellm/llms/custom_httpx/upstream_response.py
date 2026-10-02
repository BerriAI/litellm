from collections.abc import Callable, Mapping
from typing import Final

import httpx
from openai import AsyncOpenAI, OpenAI

from litellm.litellm_core_utils.upstream_response_capture import (
    record_current_upstream,
)


def record_upstream_response(response: httpx.Response) -> None:
    try:
        record_current_upstream(response.status_code, tuple(response.headers.multi_items()))
    except Exception:
        return


async def arecord_upstream_response(response: httpx.Response) -> None:
    record_upstream_response(response)


def with_capture_hooks(
    event_hooks: Mapping[str, list[Callable[..., object]]] | None,
    *,
    is_async: bool,
) -> dict[str, list[Callable[..., object]]]:
    response_hooks: Final = tuple((event_hooks or {}).get("response", ()))
    hooks: Final = {name: list(values) for name, values in (event_hooks or {}).items()}
    capture_hook: Final = arecord_upstream_response if is_async else record_upstream_response
    existing_hooks: Final = tuple(
        hook
        for hook in response_hooks
        if hook is not record_upstream_response and hook is not arecord_upstream_response
    )
    return {**hooks, "response": [capture_hook, *existing_hooks]}


def install_capture_hook(client: httpx.Client | httpx.AsyncClient) -> None:
    is_async: Final = isinstance(client, httpx.AsyncClient)
    capture_hook: Final = arecord_upstream_response if is_async else record_upstream_response
    if capture_hook in client.event_hooks.get("response", []):
        return
    client.event_hooks = with_capture_hooks(client.event_hooks, is_async=is_async)


def install_openai_capture_hook(client: OpenAI | AsyncOpenAI) -> None:
    try:
        borrowed: Final[object] = client._client  # pyright: ignore[reportPrivateUsage]  # SDK exposes no HTTP client accessor
    except AttributeError:
        return
    if isinstance(borrowed, (httpx.Client, httpx.AsyncClient)):
        install_capture_hook(borrowed)
