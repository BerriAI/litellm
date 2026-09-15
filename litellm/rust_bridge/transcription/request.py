from __future__ import annotations

from io import BytesIO
from typing import Final, cast  # noqa: TID251  # exact built-ins are narrowed at admission

from litellm.rust_bridge.configuration import CapabilityContext

_PARAMETERS: Final = (
    "model",
    "file",
    "language",
    "prompt",
    "response_format",
    "timestamp_granularities",
    "temperature",
    "user",
    "timeout",
    "api_key",
    "api_base",
    "api_version",
    "max_retries",
    "custom_llm_provider",
)


def _format(file: object) -> str:
    if type(file) in (bytes, bytearray, BytesIO):
        return "wav"
    if type(file) is tuple and len(cast(tuple[object, ...], file)) >= 2:  # cast-ok: exact tuple checked first
        name: Final = cast(tuple[object, ...], file)[0]  # cast-ok: exact tuple checked first
        return name.rsplit(".", 1)[-1].lower() if type(name) is str and "." in name else "wav"
    return ""


def request(args: tuple[object, ...], kwargs: dict[str, object]) -> dict[str, object]:
    positional: Final = {  # mutable-ok: positional values are merged into the owned boundary request
        name: args[index] for index, name in enumerate(_PARAMETERS) if index < len(args)
    }
    supplied: Final = {**positional, **kwargs}  # mutable-ok: exact public arguments are snapshotted
    model: Final = supplied.get("model")
    explicit: Final = supplied.get("custom_llm_provider")
    prefix: Final = model.partition("/")[0] if type(model) is str else ""
    provider: Final = explicit if type(explicit) is str else prefix
    return {  # mutable-ok: PyO3 requires an owned exact dict at admission
        **supplied,
        "model": model,
        "custom_llm_provider": provider,
        "audio": {"format": _format(supplied.get("file"))},  # mutable-ok: early admission fact is owned
        "optional_params": {},  # mutable-ok: host projection fills parameters after admission
    }


def context(boundary_request: dict[str, object]) -> CapabilityContext:
    model: Final = boundary_request.get("model")
    provider: Final = boundary_request.get("custom_llm_provider")
    return CapabilityContext(
        provider=provider if type(provider) is str else "",
        model=model if type(model) is str else "",
    )
