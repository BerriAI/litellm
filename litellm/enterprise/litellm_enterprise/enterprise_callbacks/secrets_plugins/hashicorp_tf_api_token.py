"""
This plugin searches for HashiCorp Terraform user/org API tokens.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class HashiCorpTFApiTokenDetector(RegexBasedDetector):
    """Scans for HashiCorp Terraform User/Org API Tokens."""

    @property
    def secret_type(self) -> str:
        return "HashiCorp Terraform API Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # HashiCorp Terraform user/org API token
            re.compile(r"""(?i)[a-z0-9]{14}\.atlasv1\.[a-z0-9\-_=]{60,70}"""),
        ]
