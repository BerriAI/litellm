"""
This plugin searches for PyPI Upload Tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class PyPiUploadTokenDetector(RegexBasedDetector):
    """Scans for PyPI Upload Tokens."""

    @property
    def secret_type(self) -> str:
        return "PyPI Upload Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [re.compile(r"""pypi-AgEIcHlwaS5vcmc[A-Za-z0-9\-_]{50,1000}""")]
