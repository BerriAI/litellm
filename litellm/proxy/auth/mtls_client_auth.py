from collections.abc import Sequence
from enum import Enum
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator


class MTLSClient(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    issuer: Annotated[str, Field(min_length=1)]
    subject: Annotated[str, Field(min_length=1)]
    end_user_id: Annotated[str, Field(min_length=1)] | None = None


_CLIENTS: Final = TypeAdapter(tuple[MTLSClient, ...])


class MTLSClientAuth(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)

    issuer_header: Annotated[str, Field(pattern=r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")] = "x-amzn-mtls-clientcert-issuer"
    subject_header: Annotated[str, Field(pattern=r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")] = "x-amzn-mtls-clientcert-subject"
    allowed_clients: tuple[MTLSClient, ...]

    @field_validator("allowed_clients", mode="before")
    @classmethod
    def validate_clients(cls, value: object) -> tuple[MTLSClient, ...]:
        return _CLIENTS.validate_python(value)


class MTLSDenial(Enum):
    CERTIFICATE = "Client certificate is missing or not approved"
    SPOOFED_IDENTITY = "Client-supplied workload identity is not allowed"


class _HeaderMapping(BaseModel):
    model_config = ConfigDict(frozen=True, strict=True)

    header_name: str
    litellm_user_role: str


_HEADER_MAPPINGS: Final = TypeAdapter(tuple[_HeaderMapping, ...])


def mtls_customer_headers(mappings: object) -> tuple[str, ...]:
    if mappings is None:
        return ()
    return tuple(
        mapping.header_name.lower()
        for mapping in _HEADER_MAPPINGS.validate_python(mappings)
        if mapping.litellm_user_role == "customer"
    )


def approve_mtls_client(
    config: MTLSClientAuth,
    headers: Sequence[tuple[str, str]],
    customer_headers: tuple[str, ...] = (),
) -> str | MTLSDenial:
    forbidden: Final = frozenset(("x-litellm-customer-id", "x-litellm-end-user-id", *customer_headers))
    if any(name.lower() in forbidden for name, _ in headers):
        return MTLSDenial.SPOOFED_IDENTITY
    issuers: Final = tuple(value for name, value in headers if name.lower() == config.issuer_header.lower())
    subjects: Final = tuple(value for name, value in headers if name.lower() == config.subject_header.lower())
    if len(issuers) != 1 or len(subjects) != 1:
        return MTLSDenial.CERTIFICATE
    return next(
        (
            client.end_user_id if client.end_user_id is not None else client.subject
            for client in config.allowed_clients
            if (client.issuer, client.subject) == (issuers[0], subjects[0])
        ),
        MTLSDenial.CERTIFICATE,
    )
