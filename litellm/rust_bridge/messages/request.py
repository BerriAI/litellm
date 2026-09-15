from __future__ import annotations

from typing import Final, cast  # noqa: TID251  # global callback registry is dynamically typed

from litellm.rust_bridge.configuration import CapabilityContext, DeliveryMode
from litellm.types.llms.anthropic import ANTHROPIC_ADVISOR_TOOL_TYPE

_PARAMETERS: Final = (
    "max_tokens",
    "messages",
    "model",
    "metadata",
    "stop_sequences",
    "stream",
    "system",
    "temperature",
    "thinking",
    "tool_choice",
    "tools",
    "top_k",
    "top_p",
    "container",
    "api_key",
    "api_base",
    "client",
    "custom_llm_provider",
)
_BODY_FIELDS: Final = frozenset(
    {
        "container",
        "max_tokens",
        "messages",
        "metadata",
        "stop_sequences",
        "stream",
        "system",
        "temperature",
        "thinking",
        "tool_choice",
        "tools",
        "top_k",
        "top_p",
    }
)


def request(args: tuple[object, ...], kwargs: dict[str, object]) -> dict[str, object]:
    positional: Final = {  # mutable-ok: positional values are merged into the owned boundary request
        name: args[index] for index, name in enumerate(_PARAMETERS) if index < len(args)
    }
    supplied: Final = {**positional, **kwargs}  # mutable-ok: exact public arguments are snapshotted
    body: Final = {  # mutable-ok: the native route consumes an owned JSON body
        key: value for key, value in supplied.items() if key in _BODY_FIELDS and value is not None
    }
    return {  # mutable-ok: PyO3 requires an owned exact dict at admission
        **supplied,
        "model": supplied.get("model"),
        "body": body,
        "has_agentic_hook": _host_operations_needed(supplied),
    }


def _host_operations_needed(supplied: dict[str, object]) -> bool:
    import litellm

    callbacks: Final = cast(list[object], litellm.callbacks)  # cast-ok: global callback registry is list-backed
    if callbacks:
        return True
    tools: Final = supplied.get("tools")
    if type(tools) is not list:
        return False
    return any(
        type(tool) is dict
        and type(cast(dict[object, object], tool).get("type")) is str
        and cast(dict[object, object], tool).get("type") == ANTHROPIC_ADVISOR_TOOL_TYPE
        for tool in cast(list[object], tools)  # cast-ok: exact list checked before safe element inspection
    )


def context(boundary_request: dict[str, object]) -> CapabilityContext:
    model: Final = boundary_request.get("model")
    provider: Final = boundary_request.get("custom_llm_provider")
    body: Final = boundary_request.get("body")
    body_mapping: Final = (
        cast(dict[str, object], body)  # cast-ok: exact dict type checked immediately before narrowing
        if type(body) is dict
        else {}  # mutable-ok: empty local view is never exposed
    )
    streaming: Final = body_mapping.get("stream") is True
    return CapabilityContext(
        provider=provider if isinstance(provider, str) else "",
        model=model if isinstance(model, str) else "",
        delivery=DeliveryMode.STREAMING if streaming else DeliveryMode.COMPLETED,
    )
