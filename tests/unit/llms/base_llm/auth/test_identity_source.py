from types import MappingProxyType
from typing import Final, Literal

import pytest
from pydantic import ValidationError

from litellm.llms.base_llm.auth.identity_source import (
    AnthropicIdentitySourceKind,
    InternalIssuerSource,
    KeycloakSource,
    identity_source_config_adapter,
    identity_source_ref,
)

SIGNING_KEY_REF: Final = "oidc/env/ISSUER_SIGNING_KEY_PEM"
OTHER_SIGNING_KEY_REF: Final = "oidc/env/OTHER_SIGNING_KEY_PEM"
CLIENT_SECRET_REF: Final = "oidc/env/KEYCLOAK_CLIENT_SECRET"
ISSUER_URL: Final = "https://issuer.internal.example"
SUBJECT: Final = "workload-a"
TOKEN_URL: Final = "https://keycloak.example/realms/litellm/protocol/openid-connect/token"
CLIENT_ID: Final = "litellm"


def make_issuer(
    issuer_url: str = ISSUER_URL,
    subject: str = SUBJECT,
    signing_key_ref: str = SIGNING_KEY_REF,
    ttl_seconds: int = 300,
) -> InternalIssuerSource:
    return InternalIssuerSource(
        issuer_url=issuer_url, subject=subject, signing_key_ref=signing_key_ref, ttl_seconds=ttl_seconds
    )


def make_keycloak(
    token_url: str = TOKEN_URL,
    client_id: str = CLIENT_ID,
    client_secret_ref: str = CLIENT_SECRET_REF,
    auth_method: Literal["client_secret_basic", "client_secret_post"] = "client_secret_basic",
    scope: str | None = None,
) -> KeycloakSource:
    return KeycloakSource(
        token_url=token_url,
        client_id=client_id,
        client_secret_ref=client_secret_ref,
        auth_method=auth_method,
        scope=scope,
    )


class TestIdentitySourceRefHashing:
    def test_identical_config_hashes_idempotently(self):
        assert identity_source_ref(make_issuer()) == identity_source_ref(make_issuer())

    def test_ref_is_prefixed_by_kind(self):
        assert identity_source_ref(make_issuer()).startswith("oidc/internal_issuer/")
        assert identity_source_ref(make_keycloak()).startswith("oidc/keycloak/")

    def test_pointer_name_change_changes_ref(self):
        """Two configs differing only in which secret a pointer names must never collide, since a
        stale ref would let the token exchange's outer cache key alias two different credentials."""
        first: Final = identity_source_ref(make_issuer(signing_key_ref=SIGNING_KEY_REF))
        second: Final = identity_source_ref(make_issuer(signing_key_ref=OTHER_SIGNING_KEY_REF))

        assert first != second

    def test_non_pointer_field_change_changes_ref(self):
        first: Final = identity_source_ref(make_keycloak(scope="openid"))
        second: Final = identity_source_ref(make_keycloak(scope="openid profile"))

        assert first != second

    def test_ref_never_contains_the_pointer_field_values(self):
        """The ref is a fixed-width hash, not a serialization of the config, so no field value -
        pointer name or otherwise - can leak into the secret-free string echoed into errors."""
        ref: Final = identity_source_ref(make_issuer())

        assert SIGNING_KEY_REF not in ref
        assert "issuer.internal.example" not in ref

    def test_different_kinds_with_disjoint_fields_never_collide(self):
        assert identity_source_ref(make_issuer()) != identity_source_ref(make_keycloak())


class TestInternalIssuerSourceValidation:
    def test_defaults(self):
        source: Final = make_issuer()

        assert source.kind == AnthropicIdentitySourceKind.internal_issuer
        assert source.ttl_seconds == 300
        assert source.audience is None

    def test_ttl_seconds_over_one_hour_is_rejected(self):
        with pytest.raises(ValidationError):
            make_issuer(ttl_seconds=3601)

    def test_ttl_seconds_at_one_hour_is_accepted(self):
        assert make_issuer(ttl_seconds=3600).ttl_seconds == 3600

    def test_non_positive_ttl_seconds_is_rejected(self):
        with pytest.raises(ValidationError):
            make_issuer(ttl_seconds=0)

    def test_missing_signing_key_ref_is_rejected(self):
        missing_field: Final = MappingProxyType({"issuer_url": ISSUER_URL, "subject": SUBJECT})

        with pytest.raises(ValidationError):
            InternalIssuerSource.model_validate(missing_field)

    def test_keycloak_only_field_is_rejected_as_extra(self):
        mixed_variant: Final = MappingProxyType(
            {
                "issuer_url": ISSUER_URL,
                "subject": SUBJECT,
                "signing_key_ref": SIGNING_KEY_REF,
                "client_secret_ref": CLIENT_SECRET_REF,
            }
        )

        with pytest.raises(ValidationError):
            InternalIssuerSource.model_validate(mixed_variant)

    def test_is_frozen(self):
        source: Final = make_issuer()

        with pytest.raises(ValidationError):
            source.subject = "workload-b"

    def test_secret_pasted_into_wrong_typed_field_is_not_echoed_in_the_error(self):
        """hide_input_in_errors keeps a value the operator pasted into a mistyped field out of the
        validation error, so a client_secret headed for the wrong field isn't logged in the raise."""
        leaked_secret: Final = "shh-do-not-log-me"
        wrong_type: Final = MappingProxyType(
            {
                "issuer_url": ISSUER_URL,
                "subject": SUBJECT,
                "signing_key_ref": SIGNING_KEY_REF,
                "ttl_seconds": leaked_secret,
            }
        )

        with pytest.raises(ValidationError) as exc_info:
            InternalIssuerSource.model_validate(wrong_type)

        assert leaked_secret not in str(exc_info.value)


class TestKeycloakSourceValidation:
    def test_defaults(self):
        source: Final = make_keycloak()

        assert source.kind == AnthropicIdentitySourceKind.keycloak
        assert source.auth_method == "client_secret_basic"
        assert source.scope is None

    def test_client_secret_post_is_accepted(self):
        assert make_keycloak(auth_method="client_secret_post").auth_method == "client_secret_post"

    def test_private_key_jwt_is_not_a_supported_auth_method_yet(self):
        unshipped_auth_method: Final = MappingProxyType(
            {
                "token_url": TOKEN_URL,
                "client_id": CLIENT_ID,
                "client_secret_ref": CLIENT_SECRET_REF,
                "auth_method": "private_key_jwt",
            }
        )

        with pytest.raises(ValidationError):
            KeycloakSource.model_validate(unshipped_auth_method)

    def test_audience_field_was_dropped(self):
        dropped_field: Final = MappingProxyType(
            {
                "token_url": TOKEN_URL,
                "client_id": CLIENT_ID,
                "client_secret_ref": CLIENT_SECRET_REF,
                "audience": "https://anthropic.example",
            }
        )

        with pytest.raises(ValidationError):
            KeycloakSource.model_validate(dropped_field)

    def test_missing_client_secret_ref_is_rejected(self):
        missing_field: Final = MappingProxyType({"token_url": TOKEN_URL, "client_id": CLIENT_ID})

        with pytest.raises(ValidationError):
            KeycloakSource.model_validate(missing_field)


class TestDiscriminatedUnionParsing:
    def test_parses_internal_issuer_variant(self):
        parsed: Final = identity_source_config_adapter.validate_python(
            MappingProxyType(
                {
                    "kind": "internal_issuer",
                    "issuer_url": ISSUER_URL,
                    "subject": SUBJECT,
                    "signing_key_ref": SIGNING_KEY_REF,
                }
            )
        )

        assert isinstance(parsed, InternalIssuerSource)

    def test_parses_keycloak_variant(self):
        parsed: Final = identity_source_config_adapter.validate_python(
            MappingProxyType(
                {
                    "kind": "keycloak",
                    "token_url": TOKEN_URL,
                    "client_id": CLIENT_ID,
                    "client_secret_ref": CLIENT_SECRET_REF,
                }
            )
        )

        assert isinstance(parsed, KeycloakSource)

    def test_unknown_kind_is_a_hard_error(self):
        with pytest.raises(ValidationError):
            identity_source_config_adapter.validate_python(MappingProxyType({"kind": "token_file"}))

    def test_mixed_variant_fields_are_a_hard_error(self):
        """A keycloak field on an internal_issuer-tagged payload must fail closed rather than be
        silently dropped or silently accepted as if it selected the other variant."""
        with pytest.raises(ValidationError):
            identity_source_config_adapter.validate_python(
                MappingProxyType(
                    {
                        "kind": "internal_issuer",
                        "issuer_url": ISSUER_URL,
                        "subject": SUBJECT,
                        "signing_key_ref": SIGNING_KEY_REF,
                        "client_secret_ref": CLIENT_SECRET_REF,
                    }
                )
            )
