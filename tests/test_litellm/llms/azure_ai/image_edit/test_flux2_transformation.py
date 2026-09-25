import base64
import io
import tempfile
from typing import Final

from litellm.llms.azure_ai.image_edit.flux2_transformation import AzureFoundryFlux2ImageEditConfig


class _ReadOnlyStream:
    def __init__(self, content: bytes) -> None:
        self._content: Final = content

    def read(self, size: int = -1) -> bytes:
        return self._content


class _NonSeekableStream(io.BytesIO):
    def seekable(self) -> bool:
        return False


def test_convert_image_to_base64_accepts_streams_without_seekable() -> None:
    encoded: Final = AzureFoundryFlux2ImageEditConfig()._convert_image_to_base64(_ReadOnlyStream(b"png-bytes"))

    assert base64.b64decode(encoded) == b"png-bytes"


def test_convert_image_to_base64_reads_non_seekable_streams_without_rewinding() -> None:
    encoded: Final = AzureFoundryFlux2ImageEditConfig()._convert_image_to_base64(_NonSeekableStream(b"png-bytes"))

    assert base64.b64decode(encoded) == b"png-bytes"


def test_convert_image_to_base64_rewinds_seekable_streams_before_encoding() -> None:
    stream: Final = io.BytesIO(b"png-bytes")
    stream.read(4)

    encoded: Final = AzureFoundryFlux2ImageEditConfig()._convert_image_to_base64(stream)

    assert base64.b64decode(encoded) == b"png-bytes"


def test_convert_image_to_base64_sends_the_full_image_on_every_retry() -> None:
    config: Final = AzureFoundryFlux2ImageEditConfig()
    stream: Final = io.BytesIO(b"png-bytes")

    first: Final = config._convert_image_to_base64(stream)
    retry: Final = config._convert_image_to_base64(stream)

    assert base64.b64decode(retry) == base64.b64decode(first) == b"png-bytes"


def test_convert_image_to_base64_accepts_named_temporary_files() -> None:
    with tempfile.NamedTemporaryFile() as upload:
        upload.write(b"png-bytes")

        encoded: Final = AzureFoundryFlux2ImageEditConfig()._convert_image_to_base64(upload)

    assert base64.b64decode(encoded) == b"png-bytes"
