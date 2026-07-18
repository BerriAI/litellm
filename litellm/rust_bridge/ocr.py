"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Final, Protocol, Union, cast

import httpx

from litellm.rust_bridge.timeouts import timeout_to_seconds as _timeout_to_seconds

if TYPE_CHECKING:
    from litellm.rust_bridge.messages import RustAmessages, RustMessages


class RustOcrError(Exception):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class RustOcr(Protocol):
    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> dict[str, object]:
        raise NotImplementedError


class RustAocr(Protocol):
    def __call__(
        self,
        model: str,
        document: dict[str, object],
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str | None,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout_seconds: float | None,
    ) -> Awaitable[dict[str, object]]:
        raise NotImplementedError


class _Unset:
    pass


_UNSET: Final[_Unset] = _Unset()
_rust_ocr_impl: RustOcr | None = None
_rust_aocr_impl: RustAocr | None = None


def _set_rust_ocr_bridge(
    ocr: RustOcr | None | _Unset = _UNSET,
    aocr: RustAocr | None | _Unset = _UNSET,
) -> None:
    global _rust_ocr_impl, _rust_aocr_impl
    if not isinstance(ocr, _Unset):
        _rust_ocr_impl = ocr
    if not isinstance(aocr, _Unset):
        _rust_aocr_impl = aocr


def use_litellm_rust(
    enabled: bool = True,
    *,
    ocr: RustOcr | None | _Unset = _UNSET,
    aocr: RustAocr | None | _Unset = _UNSET,
    messages: RustMessages | None | _Unset = _UNSET,
    amessages: RustAmessages | None | _Unset = _UNSET,
) -> None:
    configuring_ocr = not isinstance(ocr, _Unset) or not isinstance(aocr, _Unset)
    configuring_messages = not isinstance(messages, _Unset) or not isinstance(amessages, _Unset)
    if configuring_ocr or not configuring_messages:
        if enabled:
            _set_rust_ocr_bridge(ocr=ocr, aocr=aocr)
        else:
            _set_rust_ocr_bridge(ocr=None, aocr=None)
    if not configuring_messages:
        return
    from litellm.rust_bridge.messages import set_rust_messages

    if not isinstance(messages, _Unset) and not isinstance(amessages, _Unset):
        set_rust_messages(messages=messages, amessages=amessages)
    elif not isinstance(messages, _Unset):
        set_rust_messages(messages=messages)
    elif not isinstance(amessages, _Unset):
        set_rust_messages(amessages=amessages)


def rust_ocr_enabled() -> bool:
    return True


def load_rust_ocr() -> RustOcr | None:
    if _rust_ocr_impl is not None:
        return _rust_ocr_impl
    from litellm.rust_bridge import get_native_bridge

    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None
    return cast(RustOcr, native_bridge.ocr)


def load_rust_aocr() -> RustAocr | None:
    if _rust_aocr_impl is not None:
        return _rust_aocr_impl
    from litellm.rust_bridge import get_native_bridge

    native_bridge = get_native_bridge()
    if native_bridge is None:
        return None
    return cast(RustAocr, getattr(native_bridge, "aocr", None))


def ocr(
    *,
    model: str,
    document: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    optional_params: dict[str, object],
    timeout: Union[float, httpx.Timeout] | None,
) -> dict[str, object] | None:
    rust_ocr = load_rust_ocr()
    if rust_ocr is None:
        return None
    return rust_ocr(
        model=model,
        document=document,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        optional_params=optional_params,
        timeout_seconds=_timeout_to_seconds(timeout),
    )


async def aocr(
    *,
    model: str,
    document: dict[str, object],
    api_key: str | None,
    api_base: str | None,
    custom_llm_provider: str | None,
    extra_headers: dict[str, object] | None,
    optional_params: dict[str, object],
    timeout: Union[float, httpx.Timeout] | None,
) -> dict[str, object] | None:
    rust_aocr = load_rust_aocr()
    if rust_aocr is None:
        return None
    return await rust_aocr(
        model=model,
        document=document,
        api_key=api_key,
        api_base=api_base,
        custom_llm_provider=custom_llm_provider,
        extra_headers=extra_headers,
        optional_params=optional_params,
        timeout_seconds=_timeout_to_seconds(timeout),
    )
