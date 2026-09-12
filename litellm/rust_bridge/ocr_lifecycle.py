from __future__ import annotations

from collections.abc import Awaitable, Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import Final, Protocol, cast  # noqa: TID251  # validates dynamically loaded native callables

import litellm
from litellm.llms.base_llm.ocr.transformation import OCRResponse
from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.ocr import LiteLLMOcrRequest


class NativeOcrLifecycle(Protocol):
    def __call__(
        self,
        request: LiteLLMOcrRequest,
        args: Sequence[object],
        kwargs: Mapping[str, object],
        asynchronous: bool,
    ) -> OCRResponse | Awaitable[OCRResponse]: ...


class ExceptionMapper(Protocol):
    def __call__(
        self,
        *,
        model: str,
        custom_llm_provider: str | None,
        original_exception: Exception,
        completion_kwargs: dict[str, object],
        extra_kwargs: dict[str, object],
    ) -> Exception: ...


def _binding(value: object) -> NativeOcrLifecycle | None:
    if not callable(value):
        return None
    return cast("NativeOcrLifecycle", value)  # cast-ok: callable validated at the native binding boundary


NATIVE_OCR_LIFECYCLE: Final = NativeBinding("_ocr_lifecycle", validate=_binding)


def select(request: LiteLLMOcrRequest) -> NativeOcrLifecycle | None:
    if litellm.cache is not None or request.kwargs.get("caching") or request.kwargs.get("aocr"):
        return None
    return NATIVE_OCR_LIFECYCLE.load()


def arguments(request: LiteLLMOcrRequest) -> Mapping[str, object]:
    return request.kwargs


class FileReader(Protocol):
    def __call__(self) -> object: ...


def read_file_input(file_input: object) -> tuple[bytes, str | None]:
    if isinstance(file_input, str):
        raise ValueError(
            "OCR file input does not accept bare str values. Pass bytes, a pathlib.Path, or a file-like object."
        )
    if isinstance(file_input, PathLike):
        path: Final = Path(
            cast(PathLike[str], file_input)
        )  # cast-ok: Path validates the path protocol at its consumption point
        return path.read_bytes(), path.name
    if isinstance(file_input, bytes):
        return file_input, None
    reader: Final[object] = getattr(file_input, "read", None)
    if callable(reader):
        data: Final = cast(FileReader, reader)()  # cast-ok: read is callable and its return is validated below
        encoded: Final = data.encode("utf-8") if isinstance(data, str) else data
        if not isinstance(encoded, bytes):
            raise TypeError(f"OCR file read must return bytes or str, got {type(encoded)}")
        name: Final = getattr(file_input, "name", None)
        return encoded, name if isinstance(name, str) else None
    raise ValueError(
        f"Unsupported file input type: {type(file_input)}. Expected pathlib.Path, bytes, or a file-like object."
    )


def call_azure_ad_token_provider(provider: object) -> str:
    if not callable(provider):
        raise TypeError("Azure AD token provider must be callable")
    try:
        token: Final = provider()
        if not isinstance(token, str):
            raise TypeError(f"Azure AD token must be a string, got {type(token)}")
        return token
    except TypeError:
        raise
    except Exception as error:
        raise RuntimeError(f"Failed to get Azure AD token: {error}") from error


def map_failure(error: Exception, request: LiteLLMOcrRequest, request_provider: str) -> Exception:
    mapper: Final = cast(  # cast-ok: bounded adapter for the legacy public exception mapper
        ExceptionMapper, litellm.exception_type
    )
    try:
        return mapper(
            model=request.model.removeprefix(f"{request_provider}/"),
            custom_llm_provider=request_provider,
            original_exception=error,
            completion_kwargs=dict(arguments(request)),  # mutable-ok: exception mapper requires owned kwargs
            extra_kwargs=dict(request.kwargs),  # mutable-ok: exception mapper requires owned kwargs
        )
    except Exception as public_error:
        public_error.__context__ = error
        return public_error
