from contextlib import redirect_stdout
from pathlib import Path

import pytest

from litellm.proxy.common_utils.banner import (
    LITELLM_BANNER,
    LITELLM_BANNER_ASCII,
    banner_for_encoding,
    show_banner,
)


def _banner_written_to_stdout(encoding: str, destination: Path) -> str:
    with destination.open("w", encoding=encoding, errors="strict") as stream, redirect_stdout(stream):
        show_banner()
    return destination.read_text(encoding=encoding)


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16", "cp1252", "cp437", "cp850", "ascii", "latin-1", "koi8-r"])
def test_banner_for_encoding_returns_a_banner_the_encoding_can_represent(encoding: str):
    """
    The banner we hand to stdout must always survive that stream's codec.

    Regression for the proxy exiting at startup on Windows when stdout is redirected:
    the ANSI code page there is cp1252, which has none of the box-drawing characters
    the unicode banner is drawn with.
    """
    assert banner_for_encoding(encoding).encode(encoding)


def test_banner_for_encoding_prefers_the_unicode_banner_when_the_codec_allows_it():
    assert banner_for_encoding("utf-8") == LITELLM_BANNER


def test_banner_for_encoding_degrades_when_the_codec_cannot_represent_the_unicode_banner():
    assert banner_for_encoding("cp1252") == LITELLM_BANNER_ASCII


def test_banner_for_encoding_degrades_for_a_codec_python_does_not_know():
    assert banner_for_encoding("not-a-real-codec") == LITELLM_BANNER_ASCII


def test_banner_for_encoding_keeps_the_unicode_banner_when_the_stream_has_no_encoding():
    assert banner_for_encoding(None) == LITELLM_BANNER


def test_show_banner_does_not_raise_when_stdout_cannot_encode_the_unicode_banner(tmp_path: Path):
    """
    show_banner() runs as the first statement of run_server, so anything it raises takes
    the whole proxy down before it binds a port.
    """
    written = _banner_written_to_stdout("cp1252", tmp_path / "stdout.log")

    assert LITELLM_BANNER_ASCII in written


def test_show_banner_still_writes_the_unicode_banner_to_a_utf8_stdout(tmp_path: Path):
    written = _banner_written_to_stdout("utf-8", tmp_path / "stdout.log")

    assert LITELLM_BANNER in written
