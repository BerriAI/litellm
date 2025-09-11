import re

from .base import RegexBasedDetector


class SquareOAuthDetector(RegexBasedDetector):
    """Scans for Square OAuth Secrets"""

    secret_type = "Square OAuth Secret"

    denylist = [
        re.compile(r"sq0csp-[0-9A-Za-z\\\-_]{43}"),
    ]
