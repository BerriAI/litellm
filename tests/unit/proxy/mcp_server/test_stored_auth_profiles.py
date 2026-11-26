import logging
from collections.abc import Mapping
from typing import Final, cast

import pytest
from nacl.exceptions import CryptoError

from litellm.proxy.mcp_server.auth_profiles import InvalidAuthProfiles, parse_auth_profiles
from litellm.proxy.mcp_server.stored_auth_profiles import (
    MissingSigningKey,
    ProfileEncryptionFailed,
    decrypt_auth_profiles,
    encrypt_auth_profiles,
)

_OAUTH: Final = {
    "id": "Build",
    "auth_type": "oauth2",
    "oauth2_flow": "client_credentials",
    "token_url": "https://issuer.test/token",
    "client_id": "example-app",
    "client_secret": "example-secret",
    "scopes": ["read"],
}
_STATIC: Final = {"id": "Read", "auth_type": "basic", "auth_value": "example-token"}
_KEY: Final = "example-storage-key"
_REDACTED_OAUTH: Final = {
    "id": "BUILD",
    "auth_type": "oauth2",
    "oauth2_flow": "client_credentials",
}
_NEEDS_OAUTH_CREDENTIALS: Final = InvalidAuthProfiles(
    "Profile 'Build' of MCP server 'tickets' requires client_id, client_secret"
)
_NEEDS_TOKEN: Final = InvalidAuthProfiles("Profile 'Read' of MCP server 'tickets' requires authentication_token")


def _stored(*profiles: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    stored: Final = encrypt_auth_profiles("tickets", list(profiles), None, signing_key=_KEY)
    assert isinstance(stored, tuple)
    return stored


def test_a_submitted_set_is_stored_encrypted_and_decrypts_to_the_same_profiles() -> None:
    stored: Final = _stored(_OAUTH, _STATIC)
    assert "example-app" not in repr(stored)
    assert "example-secret" not in repr(stored)
    assert "example-token" not in repr(stored)
    assert decrypt_auth_profiles("tickets", list(stored), signing_key=_KEY) == parse_auth_profiles(
        "tickets", [_OAUTH, _STATIC]
    )


def test_a_redacted_edit_keeps_the_stored_ciphertext_and_the_stored_id_spelling() -> None:
    stored: Final = _stored(_OAUTH, _STATIC)
    edited: Final = encrypt_auth_profiles(
        "tickets", [{**_REDACTED_OAUTH, "scopes": ["write"]}], stored, signing_key=_KEY
    )
    assert edited == ({**stored[0], "scopes": ["write"]},)


def test_an_edit_replaces_the_stored_set_and_an_empty_submission_clears_it() -> None:
    stored: Final = _stored(_OAUTH, _STATIC)
    edited: Final = encrypt_auth_profiles("tickets", [_REDACTED_OAUTH], stored, signing_key=_KEY)
    assert isinstance(edited, tuple) and [row["id"] for row in edited] == ["Build"]
    assert encrypt_auth_profiles("tickets", [], edited, signing_key=_KEY) == ()


def test_an_edit_merges_per_profile_so_omitted_fields_stay_and_explicit_null_clears() -> None:
    stored: Final = _stored(_OAUTH)
    assert encrypt_auth_profiles("tickets", [_REDACTED_OAUTH], stored, signing_key=_KEY) == stored
    cleared: Final = encrypt_auth_profiles("tickets", [{**_REDACTED_OAUTH, "scopes": None}], stored, signing_key=_KEY)
    assert cleared == ({**stored[0], "scopes": None},)


def test_a_redacted_static_edit_keeps_the_stored_token() -> None:
    stored: Final = _stored(_STATIC)
    assert encrypt_auth_profiles("tickets", [{"id": "READ", "auth_type": "basic"}], stored, signing_key=_KEY) == stored


@pytest.mark.parametrize("empty", ("", None))
def test_an_empty_secret_on_an_edit_keeps_the_stored_one(empty: str | None) -> None:
    stored: Final = _stored(_STATIC)
    edit: Final = {"id": "Read", "auth_type": "basic", "auth_value": empty}
    assert encrypt_auth_profiles("tickets", [edit], stored, signing_key=_KEY) == stored


@pytest.mark.parametrize("empty", ("", None))
def test_an_empty_secret_on_a_new_profile_is_refused(empty: str | None) -> None:
    submitted: Final = {"id": "Read", "auth_type": "basic", "auth_value": empty}
    assert encrypt_auth_profiles("tickets", [submitted], None, signing_key=_KEY) == _NEEDS_TOKEN


def test_rotating_one_credential_preserves_the_other_ciphertext() -> None:
    stored: Final = _stored(_OAUTH)
    edit: Final = {**_REDACTED_OAUTH, "client_secret": "replacement"}
    new: Final = encrypt_auth_profiles("tickets", [edit], stored, signing_key=_KEY)
    assert isinstance(new, tuple)
    assert new[0]["client_id"] == stored[0]["client_id"]
    assert decrypt_auth_profiles("tickets", new, signing_key=_KEY) == parse_auth_profiles(
        "tickets", [{**_OAUTH, "client_secret": "replacement"}]
    )


def test_a_redacted_edit_that_resends_the_same_token_endpoint_keeps_the_credentials() -> None:
    stored: Final = _stored(_OAUTH)
    redacted: Final = {key: value for key, value in _OAUTH.items() if key not in ("client_id", "client_secret")}
    assert encrypt_auth_profiles("tickets", [redacted], stored, signing_key=_KEY) == stored


@pytest.mark.parametrize(
    ("change", "refusal"),
    [
        ({**_REDACTED_OAUTH, "token_url": "https://other.test"}, _NEEDS_OAUTH_CREDENTIALS),
        (
            {**_REDACTED_OAUTH, "token_url": None},
            InvalidAuthProfiles("Profile 'Build' of MCP server 'tickets' requires client_id, client_secret, token_url"),
        ),
        ({"id": "Read", "auth_type": "token"}, _NEEDS_TOKEN),
        (
            {"id": "Build", "auth_type": "basic"},
            InvalidAuthProfiles("Profile 'Build' of MCP server 'tickets' requires authentication_token"),
        ),
        (
            {"id": "Read", "auth_type": "oauth2", "oauth2_flow": "client_credentials"},
            InvalidAuthProfiles("Profile 'Read' of MCP server 'tickets' requires client_id, client_secret, token_url"),
        ),
    ],
)
def test_a_new_token_endpoint_or_auth_type_requires_fresh_credentials(
    change: Mapping[str, object], refusal: InvalidAuthProfiles
) -> None:
    stored: Final = _stored(_OAUTH, _STATIC)
    assert encrypt_auth_profiles("tickets", [change], stored, signing_key=_KEY) == refusal


def test_changing_issuer_with_complete_credentials_drops_the_old_issuer_settings() -> None:
    stored: Final = _stored(_OAUTH)
    update: Final = {
        **_REDACTED_OAUTH,
        "token_url": "https://new.test/token",
        "client_id": "new-app",
        "client_secret": "new-secret",
    }
    edited: Final = encrypt_auth_profiles("tickets", [update], stored, signing_key=_KEY)
    assert decrypt_auth_profiles("tickets", edited, signing_key=_KEY) == parse_auth_profiles(
        "tickets", [{**update, "id": "Build"}]
    )


def test_a_credential_the_current_key_cannot_read_is_not_kept_across_a_redacted_edit() -> None:
    stored: Final = _stored(_OAUTH)
    assert (
        encrypt_auth_profiles("tickets", [_REDACTED_OAUTH], stored, signing_key="rotated-key")
        == _NEEDS_OAUTH_CREDENTIALS
    )


def test_a_credential_that_is_not_valid_text_is_refused_as_invalid() -> None:
    submitted: Final = {**_STATIC, "auth_value": "a\ud800b"}
    assert encrypt_auth_profiles("tickets", [submitted], None, signing_key=_KEY) == InvalidAuthProfiles(
        "Profile 'Read' of MCP server 'tickets' has a credential that is not valid text"
    )


def test_a_signing_key_that_is_not_valid_text_is_a_provider_failure_not_a_bad_credential() -> None:
    assert encrypt_auth_profiles("tickets", [_STATIC], None, signing_key="key\udc80") == ProfileEncryptionFailed()


def test_an_absent_signing_key_refuses_a_write_that_needs_one_but_not_one_that_clears_the_set() -> None:
    assert encrypt_auth_profiles("tickets", [_STATIC], None, signing_key="") == MissingSigningKey()
    assert encrypt_auth_profiles("tickets", [], _stored(_STATIC), signing_key="") == ()


@pytest.mark.usefixtures("fips_mode")
def test_a_salt_key_unfit_for_fips_refuses_the_whole_edit() -> None:
    assert (
        encrypt_auth_profiles("tickets", [_STATIC, _OAUTH], None, signing_key="too-short") == ProfileEncryptionFailed()
    )


@pytest.mark.parametrize(
    "failure", (CryptoError("unavailable"), OSError("entropy unavailable"), ValueError("rejected"))
)
def test_a_provider_failure_on_a_later_credential_returns_no_rows_and_logs_only_its_class(
    failure: Exception, caplog: pytest.LogCaptureFixture
) -> None:
    def encrypt_first_only(value: str, signing_key: str) -> str:
        if value == "second-token":
            raise failure
        return f"sealed:{value}"

    second: Final = {**_STATIC, "id": "Other", "auth_value": "second-token"}
    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        result: Final = encrypt_auth_profiles(
            "tickets", [_STATIC, second], None, signing_key=_KEY, encrypt=encrypt_first_only
        )
    assert result == ProfileEncryptionFailed()
    assert [record.getMessage() for record in caplog.records] == [
        f"MCP server tickets: profile credentials were not encrypted: {type(failure).__name__}"
    ]


def test_rows_that_do_not_parse_after_encryption_are_refused() -> None:
    def encrypt_to_a_number(value: str, signing_key: str) -> str:
        return cast(str, 7)  # cast-ok: a non-text ciphertext is the input that reaches the final re-parse

    assert encrypt_auth_profiles(
        "tickets", [_STATIC], None, signing_key=_KEY, encrypt=encrypt_to_a_number
    ) == InvalidAuthProfiles(
        "additional_auth[0].authentication_token of MCP server 'tickets' is invalid: Input should be a valid string"
    )


def test_a_server_id_with_line_breaks_is_logged_on_one_line(caplog: pytest.LogCaptureFixture) -> None:
    def refuse(value: str, signing_key: str) -> str:
        raise ValueError("rejected")

    with caplog.at_level(logging.WARNING, logger="LiteLLM Proxy"):
        result: Final = encrypt_auth_profiles("tick\r\nets", [_STATIC], None, signing_key=_KEY, encrypt=refuse)
    assert result == ProfileEncryptionFailed()
    assert [record.getMessage() for record in caplog.records] == [
        "MCP server tickets: profile credentials were not encrypted: ValueError"
    ]


@pytest.mark.parametrize(
    "stored", ([{"id": "Read", "auth_type": "future_type"}], {"unexpected": 1}), ids=("unknown-type", "not-a-list")
)
def test_a_stored_set_that_cannot_be_read_inherits_nothing_but_does_not_block_a_complete_write(
    stored: object,
) -> None:
    replaced: Final = encrypt_auth_profiles("tickets", [_STATIC], stored, signing_key=_KEY)
    assert decrypt_auth_profiles("tickets", replaced, signing_key=_KEY) == parse_auth_profiles("tickets", [_STATIC])
    assert (
        encrypt_auth_profiles("tickets", [{"id": "Read", "auth_type": "basic"}], stored, signing_key=_KEY)
        == _NEEDS_TOKEN
    )
    assert encrypt_auth_profiles("tickets", [], stored, signing_key=_KEY) == ()


_NOT_A_LIST: Final = "additional_auth of MCP server 'tickets' must be a list"


@pytest.mark.parametrize(
    ("submitted", "message"),
    (
        (False, _NOT_A_LIST),
        (None, _NOT_A_LIST),
        (
            [{"id": "Read", "auth_type": "future_type"}],
            "additional_auth[0] of MCP server 'tickets' is invalid: Unknown auth type",
        ),
    ),
)
def test_a_malformed_submission_is_refused(submitted: object, message: str) -> None:
    assert encrypt_auth_profiles("tickets", submitted, None, signing_key=_KEY) == InvalidAuthProfiles(message)


def test_a_wrong_key_keeps_the_profile_listed_but_without_its_credential() -> None:
    stored: Final = _stored(_STATIC)
    assert decrypt_auth_profiles("tickets", stored, signing_key="different-key") == parse_auth_profiles(
        "tickets", [{"id": "Read", "auth_type": "basic", "authentication_token": None}]
    )


def test_an_empty_key_keeps_every_profile_listed_without_credentials() -> None:
    stored: Final = _stored(_OAUTH, _STATIC)
    assert decrypt_auth_profiles("tickets", stored, signing_key="") == parse_auth_profiles(
        "tickets",
        [
            {**_OAUTH, "client_id": None, "client_secret": None},
            {"id": "Read", "auth_type": "basic", "authentication_token": None},
        ],
    )


def test_one_damaged_ciphertext_keeps_readable_fields_and_other_profiles() -> None:
    stored: Final = _stored(_OAUTH, _STATIC)
    damaged: Final = ({**stored[0], "client_secret": "bad-ciphertext"}, stored[1])
    assert decrypt_auth_profiles("tickets", damaged, signing_key=_KEY) == parse_auth_profiles(
        "tickets", [{**_OAUTH, "client_secret": None}, _STATIC]
    )


def test_a_column_with_no_profiles_decrypts_to_no_profiles() -> None:
    assert decrypt_auth_profiles("tickets", None, signing_key=_KEY) == ()


def test_a_malformed_stored_set_refuses_the_read() -> None:
    assert decrypt_auth_profiles("tickets", False, signing_key=_KEY) == InvalidAuthProfiles(
        "additional_auth of MCP server 'tickets' must be a list"
    )
