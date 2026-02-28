from typing_extensions import TypedDict


class RedactedDict(dict):
    """Dict subclass with redacted str/repr to prevent leaking in logs."""

    def __repr__(self) -> str:
        return "RedactedDict(REDACTED)"

    def __str__(self) -> str:
        return "RedactedDict(REDACTED)"


class SecretFields(TypedDict):
    """
    Stored in data["secret_fields"]

    these fields are not logged, but used for internal purposes.
    """

    raw_headers: dict
