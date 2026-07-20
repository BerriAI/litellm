"""Thin Python wrapper for the native Rust OCR bridge."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Awaitable, Final, Protocol, Union, cast

import httpx

from litellm.rust_bridge.timeouts import timeout_to_seconds as _timeout_to_seconds

if TYPE_CHECKING:
    from litellm.rust_bridge.messages import RustAmessages, RustMessages


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


def _env_enables_rust_ocr() -> bool:
    return os.getenv("LITELLM_USE_RUST_OCR", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


_rust_ocr_enabled = _env_enables_rust_ocr()
_rust_ocr_impl: RustOcr | None = None
_rust_aocr_impl: RustAocr | None = None


def use_litellm_rust(
    enabled: bool = True,
    *,
    ocr: RustOcr | None | _Unset = _UNSET,
    aocr: RustAocr | None | _Unset = _UNSET,
    messages: RustMessages | None | _Unset = _UNSET,
    amessages: RustAmessages | None | _Unset = _UNSET,
    responses_websocket: Any | None | _Unset = _UNSET,
) -> None:
    global _rust_ocr_enabled, _rust_ocr_impl, _rust_aocr_impl
    configuring_ocr = not isinstance(ocr, _Unset) or not isinstance(aocr, _Unset)
    configuring_messages = not isinstance(messages, _Unset) or not isinstance(amessages, _Unset)
    configuring_responses_websocket = not isinstance(responses_websocket, _Unset)
    if configuring_ocr or (not configuring_messages and not configuring_responses_websocket):
        _rust_ocr_enabled = enabled
    if not isinstance(ocr, _Unset):
        _rust_ocr_impl = ocr
    if not isinstance(aocr, _Unset):
        _rust_aocr_impl = aocr
    if not configuring_messages and not configuring_responses_websocket:
        return
    if configuring_messages:
        from litellm.rust_bridge.messages import set_rust_messages

        if not isinstance(messages, _Unset) and not isinstance(amessages, _Unset):
            set_rust_messages(messages=messages, amessages=amessages)
        elif not isinstance(messages, _Unset):
            set_rust_messages(messages=messages)
        else:
            set_rust_messages(amessages=amessages)
    if configuring_responses_websocket:
        from litellm.rust_bridge.responses_websocket import set_rust_responses_websocket

        set_rust_responses_websocket(connection=responses_websocket)


def rust_ocr_enabled() -> bool:
    return _rust_ocr_enabled


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
