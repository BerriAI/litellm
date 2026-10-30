import json
from typing import Final

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from litellm.proxy._types import GenerateKeyRequest, NewTeamRequest, UpdateKeyRequest, UpdateTeamRequest, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import enforce_native_mcp_policy_is_admin_only
from litellm.proxy.management_endpoints.common_utils import update_metadata_fields
from litellm.proxy.management_endpoints.key_management_endpoints import prepare_metadata_fields

URL: Final = "https://mcp.example.com/github"


def policy(tools: object = ("search_code",), url: str = URL) -> dict[str, object]:
    return {"mcp_servers": [{"name": "github", "url": url, "tools": tools}]}


@pytest.mark.parametrize(
    "model,extra",
    [
        (GenerateKeyRequest, {}),
        (UpdateKeyRequest, {"key": "test"}),
        (NewTeamRequest, {}),
        (UpdateTeamRequest, {"team_id": "test"}),
    ],
)
def test_management_models_validate_top_level_and_metadata_policy(model: type, extra: dict[str, str]) -> None:
    invalid: Final = {"mcp_servers": [{"name": "github", "url": URL, "tools": "search_code"}]}
    with pytest.raises(ValidationError):
        model(**extra, native_mcp_policy=invalid)
    with pytest.raises(ValidationError):
        model(**extra, metadata={"native_mcp_policy": invalid})
    with pytest.raises(ValidationError, match="conflicts with metadata.native_mcp_policy"):
        model(**extra, native_mcp_policy=None, metadata={"native_mcp_policy": policy(["search_code"])})


def test_top_level_key_policy_updates_preserve_other_metadata_and_allow_removal() -> None:
    change: Final = UpdateKeyRequest(key="test", native_mcp_policy=policy(["search_code"]))
    stored: Final = prepare_metadata_fields(change, {}, {"other": "kept"})
    assert stored["metadata"]["other"] == "kept"
    assert json.loads(json.dumps(stored["metadata"])) == {"other": "kept", "native_mcp_policy": policy(["search_code"])}
    removed: Final = prepare_metadata_fields(
        UpdateKeyRequest(key="test", native_mcp_policy=None), {}, stored["metadata"]
    )
    assert removed["metadata"] == {"other": "kept", "native_mcp_policy": None}


def test_team_metadata_plumbing_persists_empty_policy_and_explicit_removal() -> None:
    update: Final = UpdateTeamRequest(team_id="test", native_mcp_policy={}).model_dump(mode="json", exclude_unset=True)
    update_metadata_fields(update)
    assert update == {"team_id": "test", "metadata": {"native_mcp_policy": {"mcp_servers": []}}}
    clear: Final = UpdateTeamRequest(team_id="test", native_mcp_policy=None).model_dump(exclude_unset=True)
    update_metadata_fields(clear)
    assert clear == {"team_id": "test", "metadata": {"native_mcp_policy": None}}


@pytest.mark.parametrize(
    "change",
    [
        UpdateKeyRequest(key="test", native_mcp_policy=policy("*")),
        UpdateKeyRequest(key="test", metadata={"native_mcp_policy": policy("*")}),
        UpdateKeyRequest(key="test", metadata={}),
        UpdateKeyRequest(key="test", native_mcp_policy=None),
    ],
)
def test_key_holder_cannot_broaden_or_remove_policy(change: UpdateKeyRequest) -> None:
    with pytest.raises(HTTPException) as error:
        enforce_native_mcp_policy_is_admin_only(
            change, {"native_mcp_policy": policy(["search_code"])}, UserAPIKeyAuth(user_role="internal_user"), "key"
        )
    assert error.value.status_code == 403


@pytest.mark.parametrize("nested", (False, True))
def test_key_holder_can_preserve_previously_stored_empty_policy(nested: bool) -> None:
    change: Final = UpdateKeyRequest.model_validate(
        {"key": "test", "metadata": {"native_mcp_policy": {}, "note": "updated"}}
        if nested
        else {"key": "test", "native_mcp_policy": {}, "key_alias": "updated"}
    )
    enforce_native_mcp_policy_is_admin_only(
        change, {"native_mcp_policy": {}}, UserAPIKeyAuth(user_role="internal_user"), "key"
    )
    persisted: Final = prepare_metadata_fields(change, change.model_dump(exclude_unset=True), {"native_mcp_policy": {}})
    assert json.loads(json.dumps(persisted["metadata"]))["native_mcp_policy"] == {"mcp_servers": []}


@pytest.mark.parametrize("stored", ({}, {"mcp_servers": []}, {"invalid": True}))
@pytest.mark.parametrize("replacement", (None, policy("*")))
def test_key_holder_cannot_replace_empty_or_invalid_stored_policy(stored: object, replacement: object) -> None:
    change: Final = UpdateKeyRequest.model_validate({"key": "test", "native_mcp_policy": replacement})
    with pytest.raises(HTTPException) as error:
        enforce_native_mcp_policy_is_admin_only(
            change, {"native_mcp_policy": stored}, UserAPIKeyAuth(user_role="internal_user"), "key"
        )
    assert error.value.status_code == 403
