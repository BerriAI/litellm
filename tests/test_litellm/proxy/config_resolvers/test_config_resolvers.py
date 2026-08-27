import os

from litellm.proxy.config_resolvers._descriptors import FieldDescriptor, resolve_fields
from litellm.proxy.config_resolvers.sso import (
    SSO_FIELD_ENV_VARS,
    SSO_SECRET_FIELDS,
    resolve_sso_config,
    restore_unsubmitted_sso_secrets,
)

_D = (
    FieldDescriptor("client_id", "client_id", "CLIENT_ID"),
    FieldDescriptor("scope", "scope", "SCOPE", default="openid"),
)


def test_resolve_fields_db_wins_over_env():
    values, provenance = resolve_fields(_D, {"client_id": "from-db"}, {"CLIENT_ID": "from-env"})
    assert values["client_id"] == "from-db"
    assert provenance["client_id"] == "db"


def test_resolve_fields_blank_db_falls_back_to_env():
    values, provenance = resolve_fields(_D, {"client_id": "   "}, {"CLIENT_ID": "from-env"})
    assert values["client_id"] == "from-env"
    assert provenance["client_id"] == "env"


def test_resolve_fields_blank_everywhere_falls_to_default():
    values, provenance = resolve_fields(_D, {}, {"SCOPE": ""})
    assert values["scope"] == "openid"
    assert provenance["scope"] == "default"


def test_resolve_fields_unset_everywhere():
    values, provenance = resolve_fields(_D, {}, {})
    assert values["client_id"] is None
    assert provenance["client_id"] == "unset"


def test_resolve_fields_empty_db_absent_by_default_falls_to_env():
    # SSO semantics: a present-but-empty stored value is absent, so env wins.
    values, provenance = resolve_fields(_D, {"client_id": ""}, {"CLIENT_ID": "from-env"})
    assert values["client_id"] == "from-env"
    assert provenance["client_id"] == "env"


def test_resolve_fields_empty_db_is_explicit_clear_when_flag_set():
    # Alerting semantics: a present-but-empty stored value is an explicit clear
    # that must win over a stale env var.
    values, provenance = resolve_fields(
        _D, {"client_id": ""}, {"CLIENT_ID": "stale-env"}, empty_db_is_set=True
    )
    assert values["client_id"] == ""
    assert provenance["client_id"] == "db"


def test_sso_descriptor_mapping_is_single_sourced():
    # The write path and read path both consume this mapping; it must cover every
    # env-backed SSO field and map to the uppercase env var.
    assert SSO_FIELD_ENV_VARS["generic_client_id"] == "GENERIC_CLIENT_ID"
    assert SSO_SECRET_FIELDS == frozenset(
        {"google_client_secret", "microsoft_client_secret", "generic_client_secret"}
    )


def test_sso_descriptor_mapping_covers_saml_fields():
    # SAML config is stored and read through the same descriptor table as the
    # OAuth providers; the login path reads these env vars, so the save path must
    # map every SAML field to its uppercase env var.
    assert SSO_FIELD_ENV_VARS["saml_idp_metadata_url"] == "SAML_IDP_METADATA_URL"
    assert SSO_FIELD_ENV_VARS["saml_idp_metadata_xml"] == "SAML_IDP_METADATA_XML"
    assert SSO_FIELD_ENV_VARS["saml_sp_entity_id"] == "SAML_SP_ENTITY_ID"
    assert SSO_FIELD_ENV_VARS["saml_allow_unsolicited"] == "SAML_ALLOW_UNSOLICITED"


def test_resolve_sso_config_resolves_saml_fields():
    resolved = resolve_sso_config(
        {"saml_idp_metadata_url": "https://idp.example.com/metadata"},
        {"SAML_ALLOW_UNSOLICITED": "true"},
    )
    assert resolved.config.saml_idp_metadata_url == "https://idp.example.com/metadata"
    assert resolved.provenance["saml_idp_metadata_url"] == "db"
    assert resolved.config.saml_allow_unsolicited == "true"
    assert resolved.provenance["saml_allow_unsolicited"] == "env"


def test_resolve_sso_config_returns_unmasked_secret_and_provenance():
    # The resolver hands back plaintext; masking is the endpoint's job. If the
    # resolver masked, the login path would consume a masked secret and fail.
    resolved = resolve_sso_config(
        {"generic_client_secret": "super-secret-value"},
        {"GENERIC_CLIENT_ID": "env-id"},
    )
    assert resolved.config.generic_client_secret == "super-secret-value"
    assert resolved.provenance["generic_client_secret"] == "db"
    assert resolved.config.generic_client_id == "env-id"
    assert resolved.provenance["generic_client_id"] == "env"


def test_resolve_sso_config_parses_structured_mappings():
    resolved = resolve_sso_config(
        {
            "generic_client_id": "id",
            "role_mappings": {
                "provider": "generic",
                "group_claim": "groups",
                "default_role": "internal_user",
                "roles": {},
            },
            "team_mappings": {"team_ids_jwt_field": "teams"},
        },
        {},
    )
    assert resolved.config.role_mappings is not None
    assert resolved.config.role_mappings.group_claim == "groups"
    assert resolved.config.team_mappings is not None
    assert resolved.config.team_mappings.team_ids_jwt_field == "teams"


def test_resolve_sso_config_does_not_mutate_os_environ(monkeypatch):
    # Unlike the legacy read path, resolving must not write os.environ.
    monkeypatch.delenv("GENERIC_CLIENT_ID", raising=False)
    before = dict(os.environ)
    resolve_sso_config({"generic_client_id": "id-from-db"}, os.environ)
    assert dict(os.environ) == before
    assert "GENERIC_CLIENT_ID" not in os.environ


def test_restore_unsubmitted_sso_secrets_restores_masked_value_from_stored():
    # A masked placeholder is "unchanged": the real secret must be restored.
    sso_data = {"generic_client_secret": "ae7a****5fd4"}
    restore_unsubmitted_sso_secrets(
        sso_data=sso_data,
        submitted_fields={"generic_client_secret"},
        stored_sso_data={"generic_client_secret": "real-secret"},
        env={},
    )
    assert sso_data["generic_client_secret"] == "real-secret"


def test_restore_unsubmitted_sso_secrets_restores_omitted_field():
    # The UI only mounts the edited provider's fields, so another provider's
    # secret arrives absent on a full-document save and must be kept.
    sso_data = {"google_client_secret": None}
    restore_unsubmitted_sso_secrets(
        sso_data=sso_data,
        submitted_fields=set(),
        stored_sso_data={"google_client_secret": "real-secret"},
        env={},
    )
    assert sso_data["google_client_secret"] == "real-secret"


def test_restore_unsubmitted_sso_secrets_falls_back_to_env():
    sso_data = {"generic_client_secret": "geni****cret"}
    restore_unsubmitted_sso_secrets(
        sso_data=sso_data,
        submitted_fields={"generic_client_secret"},
        stored_sso_data=None,
        env={"GENERIC_CLIENT_SECRET": "env-secret"},
    )
    assert sso_data["generic_client_secret"] == "env-secret"


def test_restore_unsubmitted_sso_secrets_leaves_nothing_to_restore():
    sso_data = {"generic_client_secret": None}
    restore_unsubmitted_sso_secrets(
        sso_data=sso_data,
        submitted_fields=set(),
        stored_sso_data=None,
        env={},
    )
    assert sso_data["generic_client_secret"] is None


def test_restore_unsubmitted_sso_secrets_honors_new_value():
    sso_data = {"generic_client_secret": "brand-new-secret"}
    restore_unsubmitted_sso_secrets(
        sso_data=sso_data,
        submitted_fields={"generic_client_secret"},
        stored_sso_data={"generic_client_secret": "old-secret"},
        env={},
    )
    assert sso_data["generic_client_secret"] == "brand-new-secret"


def test_restore_unsubmitted_sso_secrets_honors_explicit_clear():
    # Explicit nulls and empty strings still clear; only masks/omissions restore.
    sso_data = {"google_client_secret": None, "generic_client_secret": ""}
    restore_unsubmitted_sso_secrets(
        sso_data=sso_data,
        submitted_fields={"google_client_secret", "generic_client_secret"},
        stored_sso_data={"google_client_secret": "real-secret", "generic_client_secret": "other-secret"},
        env={},
    )
    assert sso_data["google_client_secret"] is None
    assert sso_data["generic_client_secret"] == ""


def test_restore_unsubmitted_sso_secrets_only_touches_secret_fields():
    # Non-secret fields keep full-document overwrite semantics: an omitted
    # endpoint stays a clear, since it is never masked in responses.
    sso_data = {"generic_token_endpoint": None, "generic_client_secret": "mask****ed"}
    restore_unsubmitted_sso_secrets(
        sso_data=sso_data,
        submitted_fields=set(),
        stored_sso_data={
            "generic_token_endpoint": "https://idp.example.com/token",
            "generic_client_secret": "real-secret",
        },
        env={},
    )
    assert sso_data["generic_token_endpoint"] is None
    assert sso_data["generic_client_secret"] == "real-secret"


def test_restore_unsubmitted_sso_secrets_covers_all_secret_fields():
    # Guard against a new secret field forgetting to mask or restore.
    assert SSO_SECRET_FIELDS == frozenset(
        {"google_client_secret", "microsoft_client_secret", "generic_client_secret"}
    )
