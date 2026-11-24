import json
from collections.abc import Callable
from inspect import signature
from typing import Final

import pytest
from pydantic import BaseModel, PydanticUserError, TypeAdapter, ValidationError

from litellm.types.mcp_server.auth_profile import (
    ClientCredentialsAuthProfile,
    MCPAuthProfile,
    StaticAuthProfile,
    parse_auth_profile_json,
)

PROFILES: Final[TypeAdapter[MCPAuthProfile]] = TypeAdapter(MCPAuthProfile)


@pytest.mark.parametrize(
    ("model", "payload"),
    (
        (StaticAuthProfile, '{"id":"bot","auth_type":"basic"}'),
        (ClientCredentialsAuthProfile, '{"id":"bot","auth_type":"oauth2","oauth2_flow":"client_credentials"}'),
    ),
)
def test_profile_json_accepts_the_keywords_advertised_by_the_installed_pydantic(
    model: type[StaticAuthProfile] | type[ClientCredentialsAuthProfile], payload: str
) -> None:
    supported: Final = signature(BaseModel.model_validate_json).parameters
    assert signature(model.model_validate_json).parameters.keys() == supported.keys()
    assert model.model_validate_json(payload, strict=True, by_name=True, by_alias=True).id == "bot"

    with pytest.raises(PydanticUserError):
        model.model_validate_json(payload, by_alias=False, by_name=False)


@pytest.mark.parametrize("auth_type", ("api_key", "bearer_token", "basic", "token", "authorization"))
@pytest.mark.parametrize("field", ("authentication_token", "auth_value"))
def test_static_profile_accepts_the_token_but_only_serializes_public_fields(auth_type: str, field: str) -> None:
    profile: Final = PROFILES.validate_python({"id": "bot", "auth_type": auth_type, field: "test-token"})

    assert isinstance(profile, StaticAuthProfile)
    assert profile.authentication_token is not None and profile.authentication_token.get_secret_value() == "test-token"
    assert profile.model_dump() == {"id": "bot", "auth_type": auth_type}
    assert json.loads(profile.model_dump_json()) == {"id": "bot", "auth_type": auth_type}


@pytest.mark.parametrize("profile_id", ("", "a" * 65, "Default", "de fault", "bot\n"))
def test_profile_rejects_reserved_and_invalid_ids(profile_id: str) -> None:
    with pytest.raises(ValidationError):
        PROFILES.validate_python({"id": profile_id, "auth_type": "basic"})


def test_client_credentials_profile_returns_the_flow_settings_without_the_credentials() -> None:
    public: Final = {
        "id": "bot",
        "auth_type": "oauth2",
        "oauth2_flow": "client_credentials",
        "token_url": "https://issuer.example/token",
        "scopes": ["read"],
        "token_endpoint_auth_method": "client_secret_post",
    }
    profile: Final = PROFILES.validate_python({**public, "client_id": "app-id", "client_secret": "app-secret"})

    assert isinstance(profile, ClientCredentialsAuthProfile)
    assert profile.client_id is not None and profile.client_id.get_secret_value() == "app-id"
    assert profile.client_secret is not None and profile.client_secret.get_secret_value() == "app-secret"
    assert PROFILES.dump_python(profile, mode="json") == public
    assert profile.model_dump() == {**public, "scopes": ("read",)}
    assert PROFILES.validate_python(public).model_dump(mode="json") == public


def test_resending_a_public_profile_preserves_omitted_oauth_settings() -> None:
    public: Final = {"id": "a" * 64, "auth_type": "oauth2", "oauth2_flow": "client_credentials"}
    profile: Final = PROFILES.validate_python({**public, "client_secret": "app-secret"})
    encoded: Final = PROFILES.dump_json(profile)

    assert json.loads(encoded) == public
    assert PROFILES.validate_json(encoded).model_fields_set == set(public)


def test_oauth_schema_documents_credentials_only_on_input() -> None:
    properties: Final = TypeAdapter(dict[str, object])
    input_fields: Final = properties.validate_python(
        PROFILES.json_schema(mode="validation")["$defs"]["ClientCredentialsAuthProfile"]["properties"]
    )
    output_fields: Final = properties.validate_python(
        PROFILES.json_schema(mode="serialization")["$defs"]["ClientCredentialsAuthProfile"]["properties"]
    )
    assert set(input_fields) - set(output_fields) == {"client_id", "client_secret"}
    assert {"id", "auth_type", "oauth2_flow", "scopes"} <= set(output_fields)


@pytest.mark.parametrize(
    "settings",
    (
        {"auth_type": "basic", "authentication_token": "canary", "auth_value": "canary"},
        {"client_secret": "canary"},
        {"auth_type": "canary", "client_secret": "canary"},
    ),
)
def test_profile_validation_errors_do_not_include_credential_input(settings: dict[str, str]) -> None:
    with pytest.raises(ValidationError) as error:
        PROFILES.validate_python({"id": "bot", **settings})
    assert "canary" not in str(error.value)
    assert "canary" not in error.value.json()


@pytest.mark.parametrize(
    ("model", "fields"),
    (
        (StaticAuthProfile, {"auth_type": "basic", "auth_value": "canary", "authentication_token": "canary"}),
        (ClientCredentialsAuthProfile, {"client_secret": "canary"}),
        (StaticAuthProfile, {"auth_type": "basic", "canary": True}),
    ),
)
def test_concrete_profile_errors_hide_private_input(
    model: type[StaticAuthProfile | ClientCredentialsAuthProfile], fields: dict[str, object]
) -> None:
    with pytest.raises(ValidationError) as error:
        model.model_validate({"id": "bot", **fields})
    assert "canary" not in str(error.value)
    assert "canary" not in repr(error.value.errors())
    assert "canary" not in error.value.json()


def test_round_trip_preserves_explicit_null_and_omits_unspecified_oauth_settings() -> None:
    public: Final = {"id": "bot", "auth_type": "oauth2", "oauth2_flow": "client_credentials", "scopes": None}
    assert PROFILES.validate_python(public).model_dump() == public


@pytest.mark.parametrize(
    "settings",
    (
        {"auth_type": "oauth2"},
        {"auth_type": "oauth2", "oauth2_flow": "authorization_code"},
        {"auth_type": "basic", "oauth2_flow": "client_credentials"},
    ),
)
def test_profile_rejects_an_incompatible_flow(settings: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        PROFILES.validate_python({"id": "bot", **settings})


@pytest.mark.parametrize(
    "read",
    (StaticAuthProfile.model_validate_json, ClientCredentialsAuthProfile.model_validate_json, parse_auth_profile_json),
)
def test_invalid_json_reports_a_parse_error_without_the_input(read: Callable[[str], MCPAuthProfile]) -> None:
    with pytest.raises(ValidationError) as error:
        read('{"client_secret":"canary"')
    assert "JSON" in str(error.value)
    assert "canary" not in str(error.value)
    assert "canary" not in repr(error.value.errors())
    assert "canary" not in error.value.json()


def test_invalid_mapping_keys_are_omitted_from_profile_errors() -> None:
    with pytest.raises(ValidationError) as error:
        PROFILES.validate_python({"id": "bot", "auth_type": "basic", b"canary": True})
    assert "canary" not in str(error.value)
    assert "canary" not in repr(error.value.errors())
    assert "canary" not in error.value.json()


def test_reading_profile_json_accepts_both_static_and_oauth_credentials() -> None:
    assert isinstance(parse_auth_profile_json('{"id":"test","auth_type":"basic"}'), StaticAuthProfile)
    assert isinstance(
        parse_auth_profile_json('{"id":"test","auth_type":"oauth2","oauth2_flow":"client_credentials"}'),
        ClientCredentialsAuthProfile,
    )
