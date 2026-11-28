from collections.abc import Mapping
from dataclasses import dataclass

from typing_extensions import NotRequired, ReadOnly, TypedDict


class RedactedDict(dict):
    """Dict subclass with redacted str/repr to prevent leaking in logs."""

    def __repr__(self) -> str:
        return "RedactedDict(REDACTED)"

    def __str__(self) -> str:
        return "RedactedDict(REDACTED)"

    def copy(self) -> "RedactedDict":
        return RedactedDict(super().copy())


@dataclass(frozen=True, slots=True)
class PinnedDeployment:
    deployment_id: str
    model_group: str
    custom_llm_provider: str | None
    served_model: str | None = None


class SecretFields(TypedDict):
    """
    Stored in data["secret_fields"]

    these fields are not logged, but used for internal purposes.
    """

    raw_headers: dict
    user_provider_credentials: NotRequired[ReadOnly[Mapping[str, Mapping[str, str]]]]
    user_provider_credentials_user_id: NotRequired[ReadOnly[str]]
