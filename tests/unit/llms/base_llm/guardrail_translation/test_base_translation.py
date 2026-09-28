import json
from typing import Final

from litellm.llms.base_llm.guardrail_translation.base_translation import BaseTranslation
from litellm.proxy._types import UserAPIKeyAuth

RAW_SESSION_TOKEN: Final = "cli-session-Qm7xJ2kP9sLw4vT1nR8yAa"


def _fully_populated_session_key() -> UserAPIKeyAuth:
    return UserAPIKeyAuth(
        token=RAW_SESSION_TOKEN,
        is_session_token=True,
        key_alias="cli-session-alice",
        user_id="alice",
        team_id="team-prod",
        org_id="org-1",
        metadata={"logging": [{"callback_name": "langfuse", "callback_vars": {"langfuse_secret_key": "SECRET-KEY"}}]},
        team_metadata={"logging": [{"callback_vars": {"langfuse_secret_key": "SECRET-TEAM"}}]},
        organization_metadata={"logging": [{"callback_vars": {"langfuse_secret_key": "SECRET-ORG"}}]},
        project_metadata={"logging": [{"callback_vars": {"langfuse_secret_key": "SECRET-PROJECT"}}]},
        jwt_claims={"sub": "alice", "email": "alice@corp.example", "name": "Alice Smith"},
        team_member={"user_id": "alice", "user_email": "alice@corp.example", "role": "admin"},
        config={"internal": "proxy-config"},
    )


def test_transform_emits_only_identity_never_credentials_or_callback_secrets():
    metadata = BaseTranslation.transform_user_api_key_dict_to_metadata(_fully_populated_session_key())
    serialized = json.dumps(metadata, default=str)

    assert metadata["user_api_key_alias"] == "cli-session-alice"
    assert metadata["user_api_key_key_alias"] == "cli-session-alice"
    assert metadata["user_api_key_hash"] == "cli-session-alice"
    assert metadata["user_api_key_team_id"] == "team-prod"
    assert RAW_SESSION_TOKEN not in serialized
    assert "callback_vars" not in serialized
    assert "SECRET" not in serialized
    assert "Alice Smith" not in serialized
    assert "proxy-config" not in serialized
    for dropped in (
        "user_api_key_token",
        "user_api_key_jwt_claims",
        "user_api_key_team_member",
        "user_api_key_organization_metadata",
        "user_api_key_project_metadata",
        "user_api_key_config",
    ):
        assert dropped not in metadata


def test_transform_of_no_key_is_empty():
    assert BaseTranslation.transform_user_api_key_dict_to_metadata(None) == {}
