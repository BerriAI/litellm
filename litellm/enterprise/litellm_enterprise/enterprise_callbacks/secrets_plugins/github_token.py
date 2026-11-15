"""
This plugin searches for GitHub tokens
"""

import re

from detect_secrets.plugins.base import RegexBasedDetector


class GitHubTokenCustomDetector(RegexBasedDetector):
    """Scans for GitHub tokens."""

    @property
    def secret_type(self) -> str:
        return "GitHub Token"

    @property
    def denylist(self) -> list[re.Pattern]:
        return [
            # GitHub App/Personal Access/OAuth Access/Refresh Token
            # ref. https://github.blog/2021-04-05-behind-githubs-new-authentication-token-formats/
            re.compile(r"(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9_]{36}"),
            # GitHub Fine-Grained Personal Access Token
            re.compile(r"github_pat_[0-9a-zA-Z_]{82}"),
            re.compile(r"gho_[0-9a-zA-Z]{36}"),
        ]
