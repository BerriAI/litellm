"""
This plugin searches for Shopify Access Tokens, Custom Access Tokens,
Private App Access Tokens, and Shared Secrets.
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class ShopifyDetector(RegexBasedDetector):
    """Scans for Shopify Access Tokens, Custom Access Tokens, Private App Access Tokens,
    and Shared Secrets.
    """

    @property
    def secret_type(self) -> str:
        return "Shopify Secret"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # Shopify access token
            re.compile(r"""shpat_[a-fA-F0-9]{32}"""),
            # Shopify custom access token
            re.compile(r"""shpca_[a-fA-F0-9]{32}"""),
            # Shopify private app access token
            re.compile(r"""shppa_[a-fA-F0-9]{32}"""),
            # Shopify shared secret
            re.compile(r"""shpss_[a-fA-F0-9]{32}"""),
        ]
