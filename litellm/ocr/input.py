import base64
import mimetypes
import os
import re
from io import IOBase
from typing import Final, Literal, Protocol

from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm._logging import verbose_logger

_MIME_PATTERN: Final = re.compile(r"^[\w.+-]+/[\w.+-]+$")

_MIME_TYPE_MAP: Final = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".tiff": "image/tiff",
    ".tif": "image/tiff",
    ".bmp": "image/bmp",
}


def get_mime_type(file_path: str) -> str:
    ext: Final = os.path.splitext(file_path)[1].lower()
    mime: Final = _MIME_TYPE_MAP.get(ext)
    if mime:
        return mime
    guessed, _ = mimetypes.guess_type(file_path)
    return guessed or "application/octet-stream"


class FileReader(Protocol):
    def read(self) -> bytes | str: ...


class FileDocument(TypedDict):
    type: ReadOnly[Literal["file"]]
    file: ReadOnly[bytes | os.PathLike[str] | FileReader]
    mime_type: ReadOnly[NotRequired[str]]


def convert_file_document_to_url_document(document: FileDocument) -> dict[str, str]:
    file_input: Final = document.get("file")
    if file_input is None:
        raise ValueError(
            "document with type='file' must include a 'file' field containing "
            "a pathlib.Path, file-like object, or bytes"
        )

    file_bytes: bytes
    mime_type: str = "application/octet-stream"
    file_name: str | None = None

    if isinstance(file_input, str):
        raise ValueError(
            "OCR file input does not accept bare str values. Pass bytes, "
            "a pathlib.Path, or a file-like object. To OCR a local file "
            "from a path, call open(path, 'rb') yourself."
        )
    if isinstance(file_input, os.PathLike):
        file_path: Final = str(file_input)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        mime_type = get_mime_type(file_path)
        file_name = os.path.basename(file_path)
        with open(file_path, "rb") as file:
            file_bytes = file.read()
    elif isinstance(file_input, bytes):
        file_bytes = file_input
    elif isinstance(file_input, IOBase) or hasattr(file_input, "read"):
        if hasattr(file_input, "name"):
            file_name = getattr(file_input, "name", None)
            if file_name:
                mime_type = get_mime_type(file_name)
        file_bytes = file_input.read()
        if isinstance(file_bytes, str):
            file_bytes = file_bytes.encode("utf-8")
    else:
        raise ValueError(
            f"Unsupported file input type: {type(file_input)}. Expected pathlib.Path, bytes, or a file-like object."
        )

    if not file_bytes:
        raise ValueError("File is empty or could not be read")

    if "mime_type" in document:
        mime_type = document["mime_type"]

    if not _MIME_PATTERN.match(mime_type):
        raise ValueError(f"Invalid MIME type: {mime_type}")

    base64_data: Final = base64.b64encode(file_bytes).decode("utf-8")
    data_uri: Final = f"data:{mime_type};base64,{base64_data}"

    if mime_type.startswith("image/"):
        verbose_logger.debug(
            "OCR file input: Converted file to image_url data URI (mime=%s, size=%s bytes, name=%s)",
            mime_type,
            len(file_bytes),
            file_name,
        )
        return {"type": "image_url", "image_url": data_uri}

    verbose_logger.debug(
        "OCR file input: Converted file to document_url data URI (mime=%s, size=%s bytes, name=%s)",
        mime_type,
        len(file_bytes),
        file_name,
    )
    return {"type": "document_url", "document_url": data_uri}
