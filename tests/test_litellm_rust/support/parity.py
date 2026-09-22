from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, fields, replace
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeVar

import httpx
import pytest

from litellm.llms.base_llm.ocr.transformation import OCRResponse

if TYPE_CHECKING:
    from _typeshed import DataclassInstance

VOLATILE_HEADERS: Final = frozenset({"date", "server"})


@dataclass(frozen=True, slots=True)
class Outcome:
    kind: str
    error_class: str | None = None
    text: str | None = None
    message: str | None = None
    status_code: object = None
    llm_provider: object = None
    model: object = None
    litellm_debug_info: object = None
    num_retries: object = None
    max_retries: object = None
    timeout: object = None
    response_status: int | None = None
    response_text: str | None = None
    response_headers: Mapping[str, str] | None = None
    litellm_response_headers: Mapping[str, str] | None = None
    result: object = None


def _headers(value: object) -> Mapping[str, str] | None:
    if not isinstance(value, httpx.Headers | Mapping):
        return None
    return MappingProxyType(
        {str(name).lower(): str(item) for name, item in value.items() if str(name).lower() not in VOLATILE_HEADERS}
    )


def _response(error: BaseException) -> httpx.Response | None:
    try:
        response: Final = getattr(error, "response", None)
    except RuntimeError:
        return None
    return response if isinstance(response, httpx.Response) else None


def failure_outcome(error: BaseException) -> Outcome:
    response: Final = _response(error)
    return Outcome(
        kind="error",
        error_class=f"{type(error).__module__}.{type(error).__qualname__}",
        text=str(error),
        message=getattr(error, "message", None),
        status_code=getattr(error, "status_code", None),
        llm_provider=getattr(error, "llm_provider", None),
        model=getattr(error, "model", None),
        litellm_debug_info=getattr(error, "litellm_debug_info", None),
        num_retries=getattr(error, "num_retries", None),
        max_retries=getattr(error, "max_retries", None),
        timeout=getattr(error, "timeout", None),
        response_status=None if response is None else response.status_code,
        response_text=None if response is None else response.text,
        response_headers=None if response is None else _headers(response.headers),
        litellm_response_headers=_headers(getattr(error, "litellm_response_headers", None)),
    )


def success_outcome(response: OCRResponse) -> Outcome:
    return Outcome(kind="ok", result=response.model_dump())


@dataclass(frozen=True, slots=True)
class Divergence:
    """A field that differs by design. With `normalize`, both sides are compared after it
    instead of the field being dropped."""

    field: str
    reason: str
    normalize: Callable[[object], object] | None = None


@dataclass(frozen=True, slots=True)
class Masked:
    """A field that holds a run-dependent value on both sides, compared after `mask`."""

    field: str
    reason: str
    mask: Callable[[object], object]


async def observe(call: Callable[[], Awaitable[OCRResponse]]) -> Outcome:
    try:
        return success_outcome(await call())
    except Exception as error:
        return failure_outcome(error)


async def run_on(backend: bool, monkeypatch: pytest.MonkeyPatch, call: Callable[[], Awaitable[OCRResponse]]) -> Outcome:
    monkeypatch.setenv("LITELLM_RUST", "1" if backend else "0")
    return await observe(call)


ObservedT = TypeVar("ObservedT", bound="DataclassInstance")


def without(outcome: ObservedT, divergences: tuple[Divergence, ...], masks: tuple[Masked, ...] = ()) -> ObservedT:
    compared: Final = {
        **{
            divergence.field: None
            if divergence.normalize is None
            else divergence.normalize(getattr(outcome, divergence.field))
            for divergence in divergences
        },
        **{masked.field: masked.mask(getattr(outcome, masked.field)) for masked in masks},
    }
    return replace(outcome, **compared)


def assert_parity(
    python: ObservedT, rust: ObservedT, divergences: tuple[Divergence, ...] = (), masks: tuple[Masked, ...] = ()
) -> None:
    known: Final = frozenset(divergence.field for divergence in divergences) | {masked.field for masked in masks}
    unknown: Final = known - {field.name for field in fields(python)}
    assert not unknown, f"divergence registry names unknown fields: {sorted(unknown)}"
    for divergence in divergences:
        assert getattr(python, divergence.field) != getattr(rust, divergence.field), (
            f"registered divergence on {divergence.field!r} no longer diverges; remove it: {divergence.reason}"
        )
    expected: Final = without(python, divergences, masks)
    actual: Final = without(rust, divergences, masks)
    differences: Final = {
        field.name: {"python": getattr(expected, field.name), "rust": getattr(actual, field.name)}
        for field in fields(python)
        if getattr(expected, field.name) != getattr(actual, field.name)
    }
    assert not differences, differences
