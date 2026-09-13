from collections.abc import Mapping
from os import PathLike
from typing import Final, Literal, Protocol, cast  # noqa: TID251  # native callables are validated when loaded

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.configuration import rust_ocr_enabled


class FileReader(Protocol):
    def read(self) -> bytes | str: ...


class FileDocument(TypedDict):
    type: ReadOnly[Literal["file"]]
    file: ReadOnly[bytes | PathLike[str] | FileReader]
    mime_type: ReadOnly[NotRequired[str]]


class NativeFileDocument(Protocol):
    def __call__(self, document: Mapping[str, object]) -> dict[str, str]: ...


class NativeUploadDocument(Protocol):
    def __call__(self, file_content: bytes, file_name: str | None, content_type: str | None) -> dict[str, str]: ...


class NativeMimeType(Protocol):
    def __call__(self, file_name: str) -> str: ...


_FILE_DOCUMENT: Final = NativeBinding(
    "_ocr_file_document",
    validate=lambda value: (
        cast(  # cast-ok: native export owns the callable signature
            NativeFileDocument, value
        )
        if callable(value)
        else None
    ),
)
_UPLOAD_DOCUMENT: Final = NativeBinding(
    "_ocr_upload_document",
    validate=lambda value: (
        cast(  # cast-ok: native export owns the callable signature
            NativeUploadDocument, value
        )
        if callable(value)
        else None
    ),
)
_MAX_FILE_BYTES: Final = NativeBinding(
    "_OCR_MAX_FILE_BYTES", validate=lambda value: value if isinstance(value, int) and value > 0 else None
)
_MIME_TYPE: Final = NativeBinding(
    "_ocr_mime_type",
    validate=lambda value: (
        cast(  # cast-ok: native export owns the callable signature
            NativeMimeType, value
        )
        if callable(value)
        else None
    ),
)
_PYTHON_MAX_FILE_BYTES: Final = 50 * 1024 * 1024


def get_mime_type(file_path: str) -> str:
    native: Final = _MIME_TYPE.load() if rust_ocr_enabled() else None
    if native is None:
        from litellm.ocr import legacy

        return legacy.get_mime_type(file_path)
    return native(file_path)


def get_max_file_bytes() -> int:
    limit: Final = _MAX_FILE_BYTES.load() if rust_ocr_enabled() else None
    if limit is None:
        return _PYTHON_MAX_FILE_BYTES
    return limit


def convert_file_document_to_url_document(document: FileDocument) -> dict[str, str]:
    native: Final = _FILE_DOCUMENT.load() if rust_ocr_enabled() else None
    if native is None:
        from litellm.ocr import legacy

        return legacy.convert_file_document_to_url_document(document)
    return native(document)


def convert_upload_to_url_document(
    file_content: bytes, filename: str | None, content_type: str | None
) -> dict[str, str]:
    native: Final = _UPLOAD_DOCUMENT.load() if rust_ocr_enabled() else None
    if native is None:
        from litellm.ocr import legacy

        if len(file_content) > _PYTHON_MAX_FILE_BYTES:
            raise ValueError("OCR file exceeds the size limit")
        content_mime: Final = content_type.split(";")[0].strip() if content_type else None
        mime_type: Final = (
            legacy.get_mime_type(filename)
            if filename and (not content_mime or content_mime == "application/octet-stream")
            else content_mime or "application/octet-stream"
        )
        return legacy.convert_file_document_to_url_document(
            {"type": "file", "file": file_content, "mime_type": mime_type}
        )
    return native(file_content, filename, content_type)
