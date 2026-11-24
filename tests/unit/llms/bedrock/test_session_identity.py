from typing import Final

import pytest

import litellm
from litellm.llms.bedrock.session_identity import resolve_bedrock_session_identity
from litellm.types.llms.bedrock import AwsSessionTag


@pytest.mark.parametrize("value", ["équipe 東京 42", "agent_.:/=+-@", "a" * 256])
def test_valid_identity_tags_keep_the_original_value(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setattr(litellm, "bedrock_session_tag_fields", ["user_api_key_team_alias"])
    assert resolve_bedrock_session_identity({"metadata": {"user_api_key_team_alias": value}}, None, None) == (
        ({"Key": "user_api_key_team_alias", "Value": value},),
        None,
    )


@pytest.mark.parametrize("value", ["", "a" * 257, "agent\nname", "agent!", 42, None, ["agent"]])
def test_invalid_identity_tags_are_dropped(monkeypatch: pytest.MonkeyPatch, value: object) -> None:
    monkeypatch.setattr(litellm, "bedrock_session_tag_fields", ["user_api_key_alias"])
    assert resolve_bedrock_session_identity({"metadata": {"user_api_key_alias": value}}, None, None) == (None, None)


def test_static_tags_win_case_insensitively_and_fill_the_tag_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "bedrock_session_tag_fields", ["user_api_key_alias", "user_api_key_team_alias"])
    tags: Final = (AwsSessionTag(Key="USER_API_KEY_ALIAS", Value="deployment"),)
    identity: Final = {"metadata": {"user_api_key_alias": "caller", "user_api_key_team_alias": "team"}}
    assert resolve_bedrock_session_identity(identity, tags, None) == (
        (*tags, {"Key": "user_api_key_team_alias", "Value": "team"}),
        None,
    )
    full: Final = (*tags, *(AwsSessionTag(Key=f"static-{index}", Value="deployment") for index in range(49)))
    assert resolve_bedrock_session_identity(identity, full, None) == (full, None)


@pytest.mark.parametrize("invalid", ["a", "a" * 65, "user/name", "équipe", "user name", "", None, 12])
def test_session_name_uses_the_first_valid_fallback_without_sanitizing(
    monkeypatch: pytest.MonkeyPatch, invalid: object
) -> None:
    monkeypatch.setattr(litellm, "bedrock_session_name_fields", ["user_api_key_user_id", "user_api_key_alias"])
    params: Final = {"litellm_metadata": {"user_api_key_user_id": invalid, "user_api_key_alias": "agent-42"}}
    assert resolve_bedrock_session_identity(params, None, None) == (None, "agent-42")
    assert resolve_bedrock_session_identity(params, None, "explicit") == (None, "explicit")


def test_missing_identity_and_unconfigured_settings_leave_auth_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "bedrock_session_tag_fields", None)
    monkeypatch.setattr(litellm, "bedrock_session_name_fields", None)
    assert resolve_bedrock_session_identity({"metadata": {"user_api_key_alias": "agent"}}, None, None) == (None, None)
    monkeypatch.setattr(litellm, "bedrock_session_tag_fields", ["user_api_key_alias", "spend_logs_metadata"])
    monkeypatch.setattr(litellm, "bedrock_session_name_fields", ["user_api_key_alias"])
    assert resolve_bedrock_session_identity(None, None, None) == (None, None)


def test_long_keys_and_duplicate_case_variants_do_not_produce_invalid_tags(monkeypatch: pytest.MonkeyPatch) -> None:
    long: Final = "user_api_key_" + "x" * 128
    monkeypatch.setattr(litellm, "bedrock_session_tag_fields", [long, "user_api_key_alias", "user_api_key_ALIAS"])
    params: Final = {"metadata": {long: "value", "user_api_key_alias": "first", "user_api_key_ALIAS": "second"}}
    assert resolve_bedrock_session_identity(params, None, None) == (
        ({"Key": "user_api_key_alias", "Value": "first"},),
        None,
    )


def test_caller_supplied_end_user_id_never_becomes_session_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "bedrock_session_tag_fields", ["user_api_key_end_user_id"])
    monkeypatch.setattr(litellm, "bedrock_session_name_fields", ["user_api_key_end_user_id"])
    params: Final = {"metadata": {"user_api_key_end_user_id": "caller-chosen"}}
    assert resolve_bedrock_session_identity(params, None, None) == (None, None)
