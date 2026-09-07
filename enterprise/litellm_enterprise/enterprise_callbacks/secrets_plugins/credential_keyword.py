"""
This plugin searches for credential-shaped values assigned to a credential-named key.
"""

import re
from collections.abc import Generator, Mapping
from typing import Final

from detect_secrets.plugins.keyword import KeywordDetector

_CREDENTIAL_VALUE: Final = re.compile(r"[A-Za-z0-9_.~+/-]+={0,2}")
_ENVIRONMENT_REFERENCE: Final = re.compile(r"os\.environ/\w+", re.IGNORECASE)
_ENVIRONMENT_VARIABLE_NAME: Final = re.compile(r"[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+")
_LOWERCASE_WORD_SEQUENCE: Final = re.compile(r"[a-z]+(?:[-._/][a-z]+)+")


class CredentialKeywordDetector(KeywordDetector):  # pyright: ignore[reportUntypedBaseClass]  # detect_secrets ships no type information
    """Yields the values ``KeywordDetector`` matches that are one credential-shaped token of
    at least ``minimum_length`` characters, dropping the ones that name a credential rather
    than holding one."""

    secret_type = "Credential Keyword"

    def __init__(
        self, minimum_length: int = 12, keyword_exclude: str | None = None
    ) -> None:
        if (
            not isinstance(minimum_length, int)  # pyright: ignore[reportUnnecessaryIsInstance]  # the value comes from an operator's YAML
            or minimum_length < 1
        ):
            raise ValueError(
                f"minimum_length must be a positive integer, got {minimum_length!r}"
            )
        super().__init__(keyword_exclude=keyword_exclude)
        self.minimum_length = minimum_length

    def _is_credential(self, value: str) -> bool:
        return (
            len(value) >= self.minimum_length
            and _CREDENTIAL_VALUE.fullmatch(value) is not None
            and _ENVIRONMENT_REFERENCE.fullmatch(value) is None
            and _ENVIRONMENT_VARIABLE_NAME.fullmatch(value) is None
            and _LOWERCASE_WORD_SEQUENCE.fullmatch(value) is None
        )

    def analyze_string(
        self,
        string: str,
        denylist_regex_to_group: Mapping[re.Pattern[str], int] | None = None,
    ) -> Generator[str, None, None]:
        yield from (
            value
            for value in super().analyze_string(string, denylist_regex_to_group)
            if self._is_credential(value)
        )
