import re
from base64 import b64encode

import requests

from ..constants import VerifiedResult
from .base import RegexBasedDetector


class StripeDetector(RegexBasedDetector):
    """Scans for Stripe keys."""

    secret_type = "Stripe Access Key"

    denylist = (
        # Stripe standard keys begin with sk_live and restricted with rk_live
        re.compile(r"(?:r|s)k_live_[0-9a-zA-Z]{24}"),
    )

    def verify(self, secret: str) -> VerifiedResult:  # pragma: no cover
        response = requests.get(
            "https://api.stripe.com/v1/charges",
            headers={
                "Authorization": b"Basic "
                + b64encode(
                    "{}:".format(secret).encode("utf-8"),
                ),
            },
        )

        if response.status_code == 200:
            return VerifiedResult.VERIFIED_TRUE

        # Restricted keys may be limited to certain endpoints
        if secret.startswith("rk_live"):
            return VerifiedResult.UNVERIFIED

        return VerifiedResult.VERIFIED_FALSE
