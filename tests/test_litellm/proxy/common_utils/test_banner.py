import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.abspath("../../../.."))

from litellm.proxy.common_utils.banner import show_banner


def _cp1252_stdout() -> io.TextIOWrapper:
    """A stdout that cannot encode the banner, like a Windows console on the default code page."""
    return io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")


class TestShowBanner:
    def test_show_banner_survives_a_stdout_that_cannot_encode_it(self):
        """The banner is decorative; an unencodable console must not stop the proxy from starting."""
        stream = _cp1252_stdout()

        with redirect_stdout(stream):
            show_banner()

        stream.flush()
        printed = stream.buffer.getvalue().decode("cp1252")
        assert "LiteLLM" in printed

    def test_show_banner_prints_the_full_banner_when_stdout_can_encode_it(self):
        stream = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="")

        with redirect_stdout(stream):
            show_banner()

        stream.flush()
        printed = stream.buffer.getvalue().decode("utf-8")
        assert "██" in printed
