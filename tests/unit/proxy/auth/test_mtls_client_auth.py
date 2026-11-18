from typing import Final

import pytest
from pydantic import ValidationError

from litellm.proxy._types import ConfigGeneralSettings
from litellm.proxy.auth.auth_utils import get_end_user_id_from_request_body
from litellm.proxy.auth.mtls_client_auth import (
    MTLSClientAuth,
    MTLSDenial,
    approve_mtls_client,
    mtls_customer_headers,
)

ISSUER: Final = "CN=WorkloadCA,O=Example"
SUBJECT: Final = "CN=billing-worker,O=Example"
HEADERS: Final = (("x-amzn-mtls-clientcert-issuer", ISSUER), ("x-amzn-mtls-clientcert-subject", SUBJECT))


@pytest.mark.parametrize("alias", (None, "billing-worker"))
def test_approved_identity_precedes_body_and_metadata(alias: str | None) -> None:
    config: Final = MTLSClientAuth.model_validate(
        {"allowed_clients": [{"issuer": ISSUER, "subject": SUBJECT, "end_user_id": alias}]}
    )
    approved: Final = approve_mtls_client(config, HEADERS)
    assert approved == (alias if alias is not None else SUBJECT)
    assert isinstance(approved, str)
    identity: Final = get_end_user_id_from_request_body(
        {"user": "body-user", "metadata": {"user": "metadata-user"}, "litellm_metadata": {"user": "litellm-user"}},
        trusted_end_user_id=approved,
    )
    assert identity == approved


@pytest.mark.parametrize(
    "headers",
    (
        (),
        HEADERS[:1],
        HEADERS[1:],
        ((HEADERS[0][0], ""), HEADERS[1]),
        (HEADERS[0], (HEADERS[1][0], "")),
        ((HEADERS[0][0], "CN=OtherCA"), HEADERS[1]),
        (HEADERS[0], (HEADERS[1][0], "CN=other-worker")),
        ((HEADERS[0][0], ISSUER.lower()), HEADERS[1]),
        (HEADERS[0], (HEADERS[1][0], SUBJECT + " ")),
        (*HEADERS, HEADERS[0]),
        (*HEADERS, HEADERS[1]),
        (*HEADERS, (HEADERS[1][0].upper(), SUBJECT)),
        (HEADERS[0], (HEADERS[1][0], SUBJECT + ", " + SUBJECT)),
    ),
)
def test_only_one_exact_approved_pair_is_accepted(headers: tuple[tuple[str, str], ...]) -> None:
    config: Final = MTLSClientAuth.model_validate({"allowed_clients": [{"issuer": ISSUER, "subject": SUBJECT}]})
    assert approve_mtls_client(config, headers) is MTLSDenial.CERTIFICATE


@pytest.mark.parametrize("header", ("X-LiteLLM-Customer-ID", "x-litellm-end-user-id", "X-Customer"))
def test_client_identity_headers_are_refused_even_when_empty(header: str) -> None:
    config: Final = MTLSClientAuth.model_validate({"allowed_clients": [{"issuer": ISSUER, "subject": SUBJECT}]})
    customer_headers: Final = mtls_customer_headers([{"header_name": "X-Customer", "litellm_user_role": "customer"}])
    assert approve_mtls_client(config, (*HEADERS, (header, "")), customer_headers) is MTLSDenial.SPOOFED_IDENTITY


def test_custom_header_names_are_case_insensitive_and_values_are_unchanged() -> None:
    config: Final = MTLSClientAuth.model_validate(
        {
            "issuer_header": "X-Certificate-Issuer",
            "subject_header": "X-Certificate-Subject",
            "allowed_clients": [{"issuer": ISSUER, "subject": SUBJECT}],
        }
    )
    assert (
        approve_mtls_client(config, (("x-certificate-issuer", ISSUER), ("X-CERTIFICATE-SUBJECT", SUBJECT))) == SUBJECT
    )
    assert approve_mtls_client(config, HEADERS) is MTLSDenial.CERTIFICATE


def test_issuer_and_subject_must_belong_to_the_same_allowlist_entry() -> None:
    config: Final = MTLSClientAuth.model_validate(
        {
            "allowed_clients": [
                {"issuer": ISSUER, "subject": "CN=another-worker"},
                {"issuer": "CN=another-ca", "subject": SUBJECT},
            ]
        }
    )
    assert approve_mtls_client(config, HEADERS) is MTLSDenial.CERTIFICATE


def test_empty_allowlist_denies_every_client() -> None:
    assert approve_mtls_client(MTLSClientAuth(allowed_clients=()), HEADERS) is MTLSDenial.CERTIFICATE


@pytest.mark.parametrize(
    "block",
    (
        {},
        {"allowed_clients": None},
        {"allowed_clients": [{}]},
        {"allowed_clients": [{"issuer": 123, "subject": SUBJECT}]},
        {"allowed_clients": [{"issuer": ISSUER, "subject": ""}]},
        {"allowed_clients": [{"issuer": ISSUER, "subject": SUBJECT, "end_user_id": ""}]},
        {"allowed_clients": [], "issuer_header": ""},
        {"allowed_clients": [], "subject_header": "bad\nheader"},
        {"allowed_clients": [], "unknown_field": True},
    ),
)
def test_invalid_configuration_is_rejected_by_general_settings(block: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        ConfigGeneralSettings.model_validate({"mtls_client_auth": block})


def test_settings_and_clients_are_immutable() -> None:
    settings: Final = ConfigGeneralSettings.model_validate(
        {"mtls_client_auth": {"allowed_clients": [{"issuer": ISSUER, "subject": SUBJECT}]}}
    )
    assert settings.mtls_client_auth is not None
    with pytest.raises(ValidationError, match="frozen"):
        setattr(settings.mtls_client_auth, "subject_header", "x-other")
    with pytest.raises(ValidationError, match="frozen"):
        setattr(settings.mtls_client_auth.allowed_clients[0], "end_user_id", "other-workload")


def test_disabled_feature_and_noncustomer_mappings() -> None:
    assert ConfigGeneralSettings().mtls_client_auth is None
    assert mtls_customer_headers(None) == ()
    assert mtls_customer_headers([{"header_name": "x-user", "litellm_user_role": "internal_user"}]) == ()
