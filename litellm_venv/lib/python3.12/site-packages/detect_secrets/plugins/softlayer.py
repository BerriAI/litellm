import re
from typing import List

import requests

from ..constants import VerifiedResult
from ..util.code_snippet import CodeSnippet
from .base import RegexBasedDetector


class SoftlayerDetector(RegexBasedDetector):
    """Scans for Softlayer credentials."""

    secret_type = "SoftLayer Credentials"

    # opt means optional
    sl = r"(?:softlayer|sl)(?:_|-|)(?:api|)"
    key_or_pass = r"(?:key|pwd|password|pass|token)"
    secret = r"([a-z0-9]{64})"
    denylist = [
        RegexBasedDetector.build_assignment_regex(
            prefix_regex=sl,
            secret_keyword_regex=key_or_pass,
            secret_regex=secret,
        ),
        re.compile(
            r"(?:http|https)://api.softlayer.com/soap/(?:v3|v3.1)/([a-z0-9]{64})",
            flags=re.IGNORECASE,
        ),
    ]

    def verify(  # type: ignore[override]  # noqa: F821
        self,
        secret: str,
        context: CodeSnippet,
    ) -> VerifiedResult:
        usernames = find_username(context)
        if not usernames:
            return VerifiedResult.UNVERIFIED

        for username in usernames:
            return verify_softlayer_key(username, secret)

        return VerifiedResult.VERIFIED_FALSE


def find_username(context: CodeSnippet) -> List[str]:
    # opt means optional
    username_keyword = (
        r"(?:"
        r"username|id|user|userid|user-id|user-name|"
        r"name|user_id|user_name|uname"
        r")"
    )
    username = r"(\w(?:\w|_|@|\.|-)+)"
    regex = re.compile(
        RegexBasedDetector.build_assignment_regex(
            prefix_regex=SoftlayerDetector.sl,
            secret_keyword_regex=username_keyword,
            secret_regex=username,
        ),
    )

    return [match for line in context for match in regex.findall(line)]


def verify_softlayer_key(username: str, token: str) -> VerifiedResult:
    headers = {"Content-type": "application/json"}
    try:
        response = requests.get(
            "https://api.softlayer.com/rest/v3/SoftLayer_Account.json",
            auth=(username, token),
            headers=headers,
        )
    except requests.exceptions.RequestException:
        return VerifiedResult.UNVERIFIED

    if response.status_code == 200:
        return VerifiedResult.VERIFIED_TRUE
    else:
        return VerifiedResult.VERIFIED_FALSE
