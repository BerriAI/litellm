"""
Tests for litellm.proxy.management_endpoints.service_account_endpoints

Covers the 4 endpoints:
    GET  /service-account/list         (3 status filters + owner_user_id)
    POST /service-account/request-rotation (owner-only flag flip)
    POST /service-account/approve       (creation OR rotation; key gen + activate)
    POST /service-account/reject        (creation→delete row, rotation→clear flag)

Mocks prisma_client, generate_key_helper_fn, _activate_service_account_and_block_previous_keys,
and the Slack send_alert so no network/DB is required.
"""

import os
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional

import pytest

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the repo root to the system path

from litellm.proxy._types import (
    ApproveServiceAccountRequest,
    RejectServiceAccountRequest,
    RequestRotationRequest,
    UserAPIKeyAuth,
)
from litellm.proxy.management_endpoints import service_account_endpoints as sae

MODULE = "litellm.proxy.management_endpoints.service_account_endpoints"


# ─── module-level GPG test keypair ───────────────────────────────────────────
# Generated once (RSA 4096) so creation-approve tests exercise the real pgpy
# encrypt path. _TEST_PUB_KEY is what gets stored on a SA row as `public_key`;
# _TEST_PRIV_KEY decrypts the ciphertext back to the plaintext in assertions.
def _make_test_keypair():
    import warnings

    warnings.filterwarnings("ignore")
    import pgpy
    from pgpy.constants import (
        HashAlgorithm,
        KeyFlags,
        PubKeyAlgorithm,
        SymmetricKeyAlgorithm,
    )

    key = pgpy.PGPKey.new(PubKeyAlgorithm.RSAEncryptOrSign, 4096)
    uid = pgpy.PGPUID.new("SA Test Requester", email="sa-test-requester@juspay.in")
    key.add_uid(
        uid,
        usage={KeyFlags.Sign, KeyFlags.EncryptCommunications, KeyFlags.EncryptStorage},
        hashes=[HashAlgorithm.SHA256],
        ciphers=[SymmetricKeyAlgorithm.AES256],
        compression=[],
    )
    return str(key.pubkey), key


_TEST_PUB_KEY, _TEST_PRIV_KEY = _make_test_keypair()


def _decrypt_armored(armored: str) -> str:
    """Decrypt a pgpy-encrypted ASCII-armored block with the test private key."""
    import warnings

    warnings.filterwarnings("ignore")
    import pgpy

    msg = pgpy.PGPMessage.from_blob(armored)
    return _TEST_PRIV_KEY.decrypt(msg).message


class _NS:
    """Minimal attribute-bag stand-in for a Prisma model row."""

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)


def _make_sa(
    user_id: str = "sa-1",
    owner_ids: List[str] = None,
    name: str = "my-sa",
    is_active: bool = False,
    is_key_rotation_requested: bool = False,
    use_case: str = "batch",
    requested_models: List[str] = None,
    requested_rpm_limit: int = 100,
    requested_parallel_requests_limit: int = 5,
    public_key: Any = _TEST_PUB_KEY,
    requester: str = None,
):
    """Build a fake LiteLLM_ServiceAccountTable row as a simple namespace.

    `public_key` defaults to a valid armored test key so creation-approve tests
    exercise the real pgpy encrypt path; pass public_key=None to test the
    missing-key failure case.
    """
    return _NS(
        user_id=user_id,
        owner_ids=owner_ids if owner_ids is not None else ["owner-1", "owner-2"],
        name=name,
        requested_models=requested_models if requested_models is not None else ["gpt-4"],
        use_case=use_case,
        requested_rpm_limit=requested_rpm_limit,
        requested_parallel_requests_limit=requested_parallel_requests_limit,
        is_active=is_active,
        is_key_rotation_requested=is_key_rotation_requested,
        public_key=public_key,
        requester=requester if requester is not None else "requester-1",
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _make_key(token: str = "tok-1", key_alias: str = "ka", blocked: bool = False):
    return _NS(
        token=token,
        key_alias=key_alias,
        key_name="sk-...1234",
        expires=datetime(2026, 12, 31, tzinfo=timezone.utc),
        blocked=blocked,
        spend=1.23,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _assert_approved_sa_user_row_synced(
    update_call, team_id: str, user_email: Optional[str] = None
):
    upd = update_call.call_args.kwargs
    assert upd["where"] == {"user_id": "sa-1"}
    assert upd["data"]["team_id"] == team_id
    assert upd["data"]["teams"] == {"set": [team_id]}
    assert upd["data"]["models"] == ["no-default-models"]
    if user_email is None:
        assert "user_email" not in upd["data"]
    else:
        assert upd["data"]["user_email"] == user_email


def _assert_approved_sa_token_fields(
    gen_kwargs,
    rpm_limit: int,
    max_parallel_requests: int,
    changed_by: str = "admin",
):
    assert gen_kwargs["models"] == ["all-team-models"]
    assert gen_kwargs["rpm_limit"] == rpm_limit
    assert gen_kwargs["max_parallel_requests"] == max_parallel_requests
    assert gen_kwargs["created_by"] == changed_by
    assert gen_kwargs["updated_by"] == changed_by
    assert gen_kwargs["allowed_routes"] == ["llm_api_routes"]


def _assert_service_account_added_to_team(
    mock_add_user_to_team,
    team_id: str,
    user_email: str,
):
    mock_add_user_to_team.assert_called_once()
    kwargs = mock_add_user_to_team.call_args.kwargs
    assert kwargs["user_id"] == "sa-1"
    assert kwargs["team_id"] == team_id
    assert kwargs["user_email"] == user_email
    assert kwargs["max_budget_in_team"] is None
    assert kwargs["user_role"] == "user"
    assert kwargs["user_api_key_dict"].user_id == "admin"


@pytest.fixture
def mock_prisma(mocker):
    """A MagicMock prisma_client with AsyncMock db accessors, patched onto
    litellm.proxy.proxy_server.prisma_client."""
    mock = mocker.MagicMock()
    mock.db.litellm_usertable.update = mocker.AsyncMock()
    mocker.patch("litellm.proxy.proxy_server.prisma_client", mock)
    return mock


@pytest.fixture
def mock_key_helpers(mocker):
    """Patch generate_key_helper_fn + _activate_service_account_and_block_previous_keys
    inside the service_account_endpoints module."""
    gen = mocker.AsyncMock()
    gen.return_value = {
        "user_id": "sa-1",
        "token": "sk-new-key-value",
        "token_id": "hashed-new-key",
        "expires": datetime(2026, 12, 31, tzinfo=timezone.utc),
    }
    activate = mocker.AsyncMock()
    mocker.patch(MODULE + ".generate_key_helper_fn", gen)
    mocker.patch(
        MODULE + "._activate_service_account_and_block_previous_keys", activate
    )
    return gen, activate


@pytest.fixture(autouse=True)
def mock_add_user_to_team(mocker):
    add_user_to_team = mocker.AsyncMock()
    mocker.patch(
        "litellm.proxy.management_endpoints.internal_user_endpoints._add_user_to_team",
        add_user_to_team,
    )
    return add_user_to_team


@pytest.fixture
def mock_slack(mocker):
    """Stub Slack sends so they do nothing (and capture calls)."""
    slack_instance = mocker.MagicMock()
    slack_instance.send_alert = mocker.AsyncMock()
    slack_instance.send_dm = mocker.AsyncMock()
    slack_instance.send_dm_file = mocker.AsyncMock(return_value=True)
    proxy_logging = mocker.MagicMock()
    proxy_logging.slack_alerting_instance = slack_instance
    mocker.patch(MODULE + ".proxy_logging_obj", proxy_logging, create=True)
    # The helper imports proxy_logging_obj lazily from proxy_server.
    mocker.patch(
        "litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging
    )
    return slack_instance


# ─── list ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_list_my_keys_filters_active_no_rotation(mock_prisma, mock_slack):
    from unittest.mock import AsyncMock

    sa_active = _make_sa(user_id="sa-active", is_active=True, is_key_rotation_requested=False)

    mock_prisma.db.litellm_serviceaccounttable.find_many = AsyncMock(return_value=[sa_active])
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[_make_key()])

    resp = await sae.list_service_accounts(
        status="my_keys", owner_user_id="owner-1",
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )

    call_where = mock_prisma.db.litellm_serviceaccounttable.find_many.call_args.kwargs["where"]
    assert call_where == {
        "is_active": True,
        "is_key_rotation_requested": False,
        "owner_ids": {"has": "owner-1"},
    }
    assert len(resp.service_accounts) == 1
    assert resp.service_accounts[0].user_id == "sa-active"
    assert resp.service_accounts[0].keys[0].token == "tok-1"


@pytest.mark.asyncio
async def test_list_creation_requests(mock_prisma, mock_slack):
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-pending", is_active=False, is_key_rotation_requested=False)
    mock_prisma.db.litellm_serviceaccounttable.find_many = AsyncMock(return_value=[sa])
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])

    resp = await sae.list_service_accounts(
        status="creation_requests", user_api_key_dict=UserAPIKeyAuth(user_id="admin")
    )
    where = mock_prisma.db.litellm_serviceaccounttable.find_many.call_args.kwargs["where"]
    assert where == {"is_active": False, "is_key_rotation_requested": False}
    assert resp.service_accounts[0].user_id == "sa-pending"
    assert resp.service_accounts[0].keys == []


@pytest.mark.asyncio
async def test_list_rotation_requests(mock_prisma, mock_slack):
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-rot", is_active=True, is_key_rotation_requested=True)
    mock_prisma.db.litellm_serviceaccounttable.find_many = AsyncMock(return_value=[sa])
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[_make_key()])

    resp = await sae.list_service_accounts(
        status="rotation_requests", user_api_key_dict=UserAPIKeyAuth(user_id="admin")
    )
    where = mock_prisma.db.litellm_serviceaccounttable.find_many.call_args.kwargs["where"]
    assert where == {"is_active": True, "is_key_rotation_requested": True}


@pytest.mark.asyncio
async def test_list_invalid_status_400(mock_prisma, mock_slack):
    with pytest.raises(Exception):
        await sae.list_service_accounts(
            status="bogus", user_api_key_dict=UserAPIKeyAuth(user_id="admin")
        )


# ─── request-rotation ────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_request_rotation_happy(mock_prisma, mock_slack):
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", owner_ids=["owner-1", "owner-2"], is_active=True)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()

    resp = await sae.request_service_account_rotation(
        data=RequestRotationRequest(user_id="sa-1", requested_by_user_id="owner-1"),
        user_api_key_dict=UserAPIKeyAuth(user_id="owner-1"),
    )
    assert resp["success"] is True
    mock_prisma.db.litellm_serviceaccounttable.update.assert_called_once()
    update_kwargs = mock_prisma.db.litellm_serviceaccounttable.update.call_args.kwargs
    assert update_kwargs["where"] == {"user_id": "sa-1"}
    assert update_kwargs["data"] == {"is_key_rotation_requested": True}


@pytest.mark.asyncio
async def test_request_rotation_non_owner_403(mock_prisma, mock_slack):
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", owner_ids=["owner-1", "owner-2"], is_active=True)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)

    with pytest.raises(Exception) as e:
        await sae.request_service_account_rotation(
            data=RequestRotationRequest(user_id="sa-1", requested_by_user_id="stranger"),
            user_api_key_dict=UserAPIKeyAuth(user_id="stranger"),
        )
    assert "403" in str(e.value) or "owner" in str(e.value).lower()


@pytest.mark.asyncio
async def test_request_rotation_not_active_409(mock_prisma, mock_slack):
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", owner_ids=["owner-1"], is_active=False)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    with pytest.raises(Exception):
        await sae.request_service_account_rotation(
            data=RequestRotationRequest(user_id="sa-1", requested_by_user_id="owner-1"),
            user_api_key_dict=UserAPIKeyAuth(user_id="owner-1"),
        )


@pytest.mark.asyncio
async def test_request_rotation_already_requested_409(mock_prisma, mock_slack):
    from unittest.mock import AsyncMock

    sa = _make_sa(
        user_id="sa-1", owner_ids=["owner-1"], is_active=True, is_key_rotation_requested=True
    )
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    with pytest.raises(Exception):
        await sae.request_service_account_rotation(
            data=RequestRotationRequest(user_id="sa-1", requested_by_user_id="owner-1"),
            user_api_key_dict=UserAPIKeyAuth(user_id="owner-1"),
        )


@pytest.mark.asyncio
async def test_request_rotation_not_found_404(mock_prisma, mock_slack):
    from unittest.mock import AsyncMock

    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=None)
    with pytest.raises(Exception):
        await sae.request_service_account_rotation(
            data=RequestRotationRequest(user_id="missing", requested_by_user_id="owner-1"),
            user_api_key_dict=UserAPIKeyAuth(user_id="owner-1"),
        )


# ─── approve ─────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_approve_creation_pending(
    mock_prisma, mock_key_helpers, mock_slack, mock_add_user_to_team
):
    from unittest.mock import AsyncMock

    gen, activate = mock_key_helpers
    sa = _make_sa(user_id="sa-1", name="my-sa", is_active=False, is_key_rotation_requested=False)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)

    resp = await sae.approve_service_account(
        data=ApproveServiceAccountRequest(user_id="sa-1", duration="30d", team_id="team-1"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )
    assert resp.user_id == "sa-1"
    # The plaintext key is GPG-encrypted to the SA's stored public key — the
    # response carries the armored ciphertext, not the raw sk- value.
    assert resp.key.startswith("-----BEGIN PGP MESSAGE-----")
    assert _decrypt_armored(resp.key) == "sk-new-key-value"
    assert resp.key_id == "hashed-new-key"
    # generate_key_helper_fn called with request_type="key" and the SA's user_id
    gen.assert_called_once()
    gen_kwargs = gen.call_args.kwargs
    assert gen_kwargs["request_type"] == "key"
    assert gen_kwargs["user_id"] == "sa-1"
    assert gen_kwargs["duration"] == "30d"
    assert gen_kwargs["team_id"] == "team-1"
    assert gen_kwargs["key_alias"] == "my-sa-service-account"
    _assert_approved_sa_token_fields(
        gen_kwargs,
        rpm_limit=100,
        max_parallel_requests=5,
    )
    # activate + block-previous-keys called with the new key hash
    activate.assert_called_once()
    assert activate.call_args.kwargs["user_id"] == "sa-1"
    assert activate.call_args.kwargs["new_key_token"] == "hashed-new-key"
    mock_prisma.db.litellm_usertable.update.assert_called_once()
    _assert_approved_sa_user_row_synced(
        mock_prisma.db.litellm_usertable.update,
        team_id="team-1",
    )
    _assert_service_account_added_to_team(
        mock_add_user_to_team,
        team_id="team-1",
        user_email="my-sa-service-account@juspay.in",
    )


@pytest.mark.asyncio
async def test_approve_creation_applies_approver_edits(
    mock_prisma, mock_key_helpers, mock_slack, mock_add_user_to_team
):
    """On creation approval, approver-provided editable fields overwrite the SA row."""
    from unittest.mock import AsyncMock

    gen, activate = mock_key_helpers
    sa = _make_sa(
        user_id="sa-1",
        name="old-name",
        use_case="old-use-case",
        requested_models=["old-model"],
        requested_rpm_limit=50,
        requested_parallel_requests_limit=2,
        owner_ids=["owner-1", "owner-2"],
        is_active=False,
        is_key_rotation_requested=False,
    )
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()

    resp = await sae.approve_service_account(
        data=ApproveServiceAccountRequest(
            user_id="sa-1",
            duration="30d",
            team_id="team-1",
            name="new-name",
            use_case="new-use-case",
            requested_models=["gpt-4", "claude-3"],
            requested_rpm_limit=200,
            requested_parallel_requests_limit=10,
            owner_ids=["owner-1", "owner-3"],
        ),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )
    assert resp.key.startswith("-----BEGIN PGP MESSAGE-----")
    assert _decrypt_armored(resp.key) == "sk-new-key-value"
    # The SA row update was called with all edited fields.
    mock_prisma.db.litellm_serviceaccounttable.update.assert_called_once()
    upd = mock_prisma.db.litellm_serviceaccounttable.update.call_args.kwargs
    assert upd["where"] == {"user_id": "sa-1"}
    data = upd["data"]
    assert data["name"] == "new-name"
    assert data["use_case"] == "new-use-case"
    assert data["requested_models"] == ["gpt-4", "claude-3"]
    assert data["requested_rpm_limit"] == 200
    assert data["requested_parallel_requests_limit"] == 10
    assert data["owner_ids"] == ["owner-1", "owner-3"]
    # The key alias reflects the edited name.
    assert gen.call_args.kwargs["key_alias"] == "new-name-service-account"
    _assert_approved_sa_token_fields(
        gen.call_args.kwargs,
        rpm_limit=200,
        max_parallel_requests=10,
    )
    mock_prisma.db.litellm_usertable.update.assert_called_once()
    _assert_approved_sa_user_row_synced(
        mock_prisma.db.litellm_usertable.update,
        team_id="team-1",
        user_email="new-name-service-account@juspay.in",
    )
    _assert_service_account_added_to_team(
        mock_add_user_to_team,
        team_id="team-1",
        user_email="new-name-service-account@juspay.in",
    )


@pytest.mark.asyncio
async def test_approve_creation_sends_updated_details_to_slack(
    mock_prisma, mock_key_helpers, mock_slack, mocker
):
    """Creation approval sends the post-edit service-account details to Slack."""
    from unittest.mock import AsyncMock

    mocker.patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test"})
    sa = _make_sa(
        user_id="sa-1",
        name="old-name",
        use_case="old-use-case",
        requested_models=["old-model"],
        requested_rpm_limit=50,
        requested_parallel_requests_limit=2,
        owner_ids=["owner-1", "owner-2"],
        is_active=False,
        is_key_rotation_requested=False,
    )
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    mock_prisma.db.litellm_usertable.update = AsyncMock()
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(
        return_value=_NS(user_id="requester-1", user_email="requester@juspay.in")
    )
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(
        return_value=[
            _NS(user_id="owner-1", user_email="owner1@juspay.in"),
            _NS(user_id="owner-3", user_email="owner3@juspay.in"),
        ]
    )

    await sae.approve_service_account(
        data=ApproveServiceAccountRequest(
            user_id="sa-1",
            duration="30d",
            team_id="team-9",
            name="new-name",
            use_case="new-use-case",
            requested_models=["gpt-4", "claude-3"],
            requested_rpm_limit=200,
            requested_parallel_requests_limit=10,
            owner_ids=["owner-1", "owner-3"],
        ),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )

    import asyncio

    await asyncio.sleep(0)
    messages = [
        call.kwargs["message"] for call in mock_slack.send_dm_file.await_args_list
    ]
    message = next(msg for msg in messages if "creation approved" in msg)
    assert "Name: new-name" in message
    assert "Use case: new-use-case" in message
    assert "Requested models: gpt-4, claude-3" in message
    assert "Requested RPM limit: 200" in message
    assert "Requested parallel requests limit: 10" in message
    assert "Owners: owner1@juspay.in, owner3@juspay.in" in message
    assert "Status: Active" in message
    assert "Key rotation requested: No" in message
    assert "Team: team-9" in message
    assert "Key ID: hashed-new-key" in message
    assert "filename.gpg" in message
    assert mock_slack.send_dm.await_count == 0


@pytest.mark.asyncio
async def test_approve_creation_syncs_user_email_on_name_edit(
    mock_prisma, mock_key_helpers, mock_slack
):
    """When the approver edits the name on a creation-approve, the SA user
    row's user_email is re-synced to <name>-service-account@juspay.in so it
    stays equal to the new key_alias."""
    from unittest.mock import AsyncMock

    gen, activate = mock_key_helpers
    sa = _make_sa(
        user_id="sa-1",
        name="old-name",
        is_active=False,
        is_key_rotation_requested=False,
    )
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    mock_prisma.db.litellm_usertable.update = AsyncMock()

    await sae.approve_service_account(
        data=ApproveServiceAccountRequest(
            user_id="sa-1", duration="30d", team_id="team-1", name="new-name"
        ),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )
    # user table email synced to the edited name's alias email, and approval
    # scopes the SA user to the selected team only.
    mock_prisma.db.litellm_usertable.update.assert_called_once()
    _assert_approved_sa_user_row_synced(
        mock_prisma.db.litellm_usertable.update,
        team_id="team-1",
        user_email="new-name-service-account@juspay.in",
    )


@pytest.mark.asyncio
async def test_approve_creation_syncs_user_team_and_models_without_name_edit(
    mock_prisma, mock_key_helpers, mock_slack
):
    """A creation-approve with no name edit must still scope the SA user row
    to the selected team and no-default-models; the provisional email stays
    untouched because it was set at filing time."""
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", name="my-sa", is_active=False)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    mock_prisma.db.litellm_usertable.update = AsyncMock()

    await sae.approve_service_account(
        data=ApproveServiceAccountRequest(user_id="sa-1", duration="30d", team_id="team-1"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )
    mock_prisma.db.litellm_usertable.update.assert_called_once()
    _assert_approved_sa_user_row_synced(
        mock_prisma.db.litellm_usertable.update,
        team_id="team-1",
    )


@pytest.mark.asyncio
async def test_approve_rotation_does_not_touch_user_email(
    mock_prisma, mock_key_helpers, mock_slack, mock_add_user_to_team
):
    """Rotation approval ignores identity edits — it must never write the
    user table email (the name is unchanged from creation)."""
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", name="my-sa", is_active=True, is_key_rotation_requested=True)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    existing_key = _NS(token="old-hash", team_id="existing-team", blocked=False)
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[existing_key])
    mock_prisma.db.litellm_verificationtoken.update = AsyncMock()
    mock_prisma.db.litellm_usertable.update = AsyncMock()

    await sae.approve_service_account(
        data=ApproveServiceAccountRequest(user_id="sa-1", duration="90d"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )
    mock_prisma.db.litellm_usertable.update.assert_not_called()
    mock_add_user_to_team.assert_not_called()


def test_sa_slug_sanitization():
    """Name → slug normalization: lowercase, collapse non-[a-z0-9-] to '-',
    strip leading/trailing '-', fall back to user_id when empty."""
    assert sae._sanitize_sa_slug("Payments Batch Runner", "uid-1") == "payments-batch-runner"
    assert sae._sanitize_sa_slug("payments-batch-runner", "uid-1") == "payments-batch-runner"
    assert sae._sanitize_sa_slug("  Multiple   Spaces  ", "uid-1") == "multiple-spaces"
    assert sae._sanitize_sa_slug("weird/name.v2", "uid-1") == "weird-name-v2"
    assert sae._sanitize_sa_slug("---leading-trailing---", "uid-1") == "leading-trailing"
    assert sae._sanitize_sa_slug("", "uid-1") == "uid-1"
    assert sae._sanitize_sa_slug(None, "uid-1") == "uid-1"
    assert sae._sanitize_sa_slug("!!!@#$%", "uid-1") == "uid-1"


def test_sa_key_alias_and_email_suffix():
    """key_alias always ends with -service-account; email is <alias>@juspay.in."""
    assert sae._sa_key_alias("payments-batch-runner", "uid-1") == "payments-batch-runner-service-account"
    assert sae._sa_key_alias(None, "uid-1") == "uid-1-service-account"
    assert sae._sa_user_email("payments-batch-runner", "uid-1") == "payments-batch-runner-service-account@juspay.in"
    assert sae._sa_user_email(None, "uid-1") == "uid-1-service-account@juspay.in"


@pytest.mark.asyncio
async def test_approve_creation_owner_ids_below_two_400(mock_prisma, mock_key_helpers, mock_slack):
    """Approving creation with <2 owners (when the approver edits owner_ids) fails."""
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", is_active=False, owner_ids=["owner-1", "owner-2"])
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    with pytest.raises(Exception):
        await sae.approve_service_account(
            data=ApproveServiceAccountRequest(
                user_id="sa-1", duration="30d", owner_ids=["only-one"]
            ),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
        )


@pytest.mark.asyncio
async def test_approve_without_team_400(mock_prisma, mock_key_helpers, mock_slack):
    """Approving without a team_id fails — the approver must pick a team."""
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", is_active=False, owner_ids=["owner-1", "owner-2"])
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    with pytest.raises(Exception):
        await sae.approve_service_account(
            data=ApproveServiceAccountRequest(user_id="sa-1", duration="30d"),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
        )


@pytest.mark.asyncio
async def test_approve_rotation_extends_existing_key_expiry_in_place(mock_prisma, mock_key_helpers, mock_slack):
    """Rotation approval extends the existing key's `expires` in place — it
    does NOT mint a new key, block the old one, or re-reveal the secret.

    The key's token/secret, alias, and team are unchanged; only `expires` is
    updated (now + duration) and the rotation flag is cleared. The returned
    `key` is empty (not re-revealed)."""
    from unittest.mock import AsyncMock

    gen, activate = mock_key_helpers
    sa = _make_sa(user_id="sa-1", name="my-sa", is_active=True, is_key_rotation_requested=True)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    existing_key = _NS(token="existing-hash", team_id="existing-team", blocked=False,
                      expires=datetime(2026, 1, 1, tzinfo=timezone.utc))
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[existing_key])
    mock_prisma.db.litellm_verificationtoken.update = AsyncMock()

    resp = await sae.approve_service_account(
        data=ApproveServiceAccountRequest(user_id="sa-1", duration="90d"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )

    # No new key minted, no prior key blocked — the existing key is extended.
    gen.assert_not_called()
    activate.assert_not_called()

    # The existing key's expires was updated in place (by its token PK).
    mock_prisma.db.litellm_verificationtoken.update.assert_called_once()
    vt_kwargs = mock_prisma.db.litellm_verificationtoken.update.call_args.kwargs
    assert vt_kwargs["where"] == {"token": "existing-hash"}
    new_expires = vt_kwargs["data"]["expires"]
    assert isinstance(new_expires, datetime)
    # ~90 days from now (allow a few seconds of slack).
    delta = new_expires - datetime.now(timezone.utc)
    assert timedelta(days=89) < delta < timedelta(days=91)

    # The rotation flag was cleared (is_active stays True).
    mock_prisma.db.litellm_serviceaccounttable.update.assert_called_once()
    sa_kwargs = mock_prisma.db.litellm_serviceaccounttable.update.call_args.kwargs
    assert sa_kwargs["where"] == {"user_id": "sa-1"}
    assert sa_kwargs["data"] == {"is_key_rotation_requested": False}

    # The key value is NOT re-revealed on rotation; the key_id is the existing key.
    assert resp.key == ""
    assert resp.key_id == "existing-hash"
    assert resp.expires == new_expires


@pytest.mark.asyncio
async def test_approve_rotation_sends_slack_notification(
    mock_prisma, mock_key_helpers, mock_slack, mocker
):
    """Rotation approval notifies owners with the updated active/no-rotation state."""
    from unittest.mock import AsyncMock

    mocker.patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test"})
    sa = _make_sa(
        user_id="sa-1",
        name="my-sa",
        is_active=True,
        is_key_rotation_requested=True,
    )
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    existing_key = _NS(token="existing-hash", blocked=False)
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[existing_key])
    mock_prisma.db.litellm_verificationtoken.update = AsyncMock()
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(
        return_value=_NS(user_id="requester-1", user_email="requester@juspay.in")
    )
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(
        return_value=[
            _NS(user_id="owner-1", user_email="owner1@juspay.in"),
            _NS(user_id="owner-2", user_email="owner2@juspay.in"),
        ]
    )

    await sae.approve_service_account(
        data=ApproveServiceAccountRequest(user_id="sa-1", duration="90d"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )

    import asyncio

    await asyncio.sleep(0)
    messages = [
        call.kwargs["message"] for call in mock_slack.send_dm.await_args_list
    ]
    message = next(msg for msg in messages if "rotation approved" in msg)
    assert "Status: Active" in message
    assert "Key rotation requested: No" in message
    assert "Owners: owner1@juspay.in, owner2@juspay.in" in message
    assert "Key expiry extended to:" in message


@pytest.mark.asyncio
async def test_approve_rotation_never_expire_duration(mock_prisma, mock_key_helpers, mock_slack):
    """duration='-1' means the key never expires — `expires` is set to None."""
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", is_active=True, is_key_rotation_requested=True)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    existing_key = _NS(token="existing-hash", team_id="existing-team", blocked=False,
                      expires=datetime(2026, 1, 1, tzinfo=timezone.utc))
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[existing_key])
    mock_prisma.db.litellm_verificationtoken.update = AsyncMock()

    resp = await sae.approve_service_account(
        data=ApproveServiceAccountRequest(user_id="sa-1", duration="-1"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )
    vt_kwargs = mock_prisma.db.litellm_verificationtoken.update.call_args.kwargs
    assert vt_kwargs["data"] == {"expires": None}
    assert resp.expires is None


@pytest.mark.asyncio
async def test_approve_rotation_one_year_duration(mock_prisma, mock_key_helpers, mock_slack):
    """duration='1y' (the UI's "1 year" option) must parse — it previously raised
    "Unsupported duration unit" because duration_in_seconds didn't handle 'y'.
    Now it extends the key by 365 days."""
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", is_active=True, is_key_rotation_requested=True)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    existing_key = _NS(token="existing-hash", team_id="existing-team", blocked=False,
                      expires=datetime(2026, 1, 1, tzinfo=timezone.utc))
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[existing_key])
    mock_prisma.db.litellm_verificationtoken.update = AsyncMock()

    resp = await sae.approve_service_account(
        data=ApproveServiceAccountRequest(user_id="sa-1", duration="1y"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )
    new_expires = mock_prisma.db.litellm_verificationtoken.update.call_args.kwargs["data"]["expires"]
    assert isinstance(new_expires, datetime)
    delta = new_expires - datetime.now(timezone.utc)
    assert timedelta(days=364) < delta < timedelta(days=366)  # ~365 days


@pytest.mark.asyncio
async def test_approve_rotation_no_existing_key_400(mock_prisma, mock_key_helpers, mock_slack):
    """Rotation with no existing key to extend fails with 400 — there's nothing
    to extend, so the approver should reject and re-create instead."""
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", is_active=True, is_key_rotation_requested=True)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_verificationtoken.find_many = AsyncMock(return_value=[])
    with pytest.raises(Exception):
        await sae.approve_service_account(
            data=ApproveServiceAccountRequest(user_id="sa-1", duration="90d"),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
        )


@pytest.mark.asyncio
async def test_approve_not_pending_409(mock_prisma, mock_key_helpers, mock_slack):
    from unittest.mock import AsyncMock

    # active SA with no rotation requested → not in a pending state
    sa = _make_sa(user_id="sa-1", is_active=True, is_key_rotation_requested=False)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    with pytest.raises(Exception):
        await sae.approve_service_account(
            data=ApproveServiceAccountRequest(user_id="sa-1", duration="30d"),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
        )


@pytest.mark.asyncio
async def test_approve_not_found_404(mock_prisma, mock_key_helpers, mock_slack):
    from unittest.mock import AsyncMock

    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=None)
    with pytest.raises(Exception):
        await sae.approve_service_account(
            data=ApproveServiceAccountRequest(user_id="missing", duration="30d"),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
        )


@pytest.mark.asyncio
async def test_approve_slack_failure_does_not_fail(mock_prisma, mock_key_helpers, mocker):
    """A Slack misconfiguration must never fail an approve."""
    from unittest.mock import AsyncMock

    gen, activate = mock_key_helpers
    sa = _make_sa(user_id="sa-1", is_active=False, is_key_rotation_requested=False)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)

    # Slack send_alert raises; the notification helper swallows it.
    slack_instance = mocker.MagicMock()
    slack_instance.send_alert = mocker.AsyncMock(side_effect=RuntimeError("no webhook"))
    slack_instance.send_dm = mocker.AsyncMock(side_effect=RuntimeError("no bot token"))
    proxy_logging = mocker.MagicMock()
    proxy_logging.slack_alerting_instance = slack_instance
    mocker.patch("litellm.proxy.proxy_server.proxy_logging_obj", proxy_logging)

    resp = await sae.approve_service_account(
        data=ApproveServiceAccountRequest(user_id="sa-1", duration="30d", team_id="team-1"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )
    # Approve still succeeded — the key is GPG-encrypted but returned.
    assert resp.key.startswith("-----BEGIN PGP MESSAGE-----")
    assert _decrypt_armored(resp.key) == "sk-new-key-value"


# ─── reject ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_reject_creation_deletes_sa_and_user(mock_prisma, mock_slack):
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", is_active=False, is_key_rotation_requested=False)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.delete = AsyncMock()
    mock_prisma.db.litellm_usertable.delete = AsyncMock()

    resp = await sae.reject_service_account(
        data=RejectServiceAccountRequest(user_id="sa-1", reason="not needed"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )
    assert resp["success"] is True
    mock_prisma.db.litellm_serviceaccounttable.delete.assert_called_once()
    assert mock_prisma.db.litellm_serviceaccounttable.delete.call_args.kwargs["where"] == {
        "user_id": "sa-1"
    }
    mock_prisma.db.litellm_usertable.delete.assert_called_once()


@pytest.mark.asyncio
async def test_reject_creation_user_delete_failure_non_blocking(mock_prisma, mock_slack):
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", is_active=False, is_key_rotation_requested=False)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.delete = AsyncMock()
    mock_prisma.db.litellm_usertable.delete = AsyncMock(side_effect=RuntimeError("fk fail"))

    # Should not raise — SA row is the contract; orphan user cleanup is best-effort.
    resp = await sae.reject_service_account(
        data=RejectServiceAccountRequest(user_id="sa-1"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )
    assert resp["success"] is True
    mock_prisma.db.litellm_serviceaccounttable.delete.assert_called_once()


@pytest.mark.asyncio
async def test_reject_rotation_clears_flag_only(mock_prisma, mock_slack):
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", is_active=True, is_key_rotation_requested=True)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    mock_prisma.db.litellm_serviceaccounttable.delete = AsyncMock()

    resp = await sae.reject_service_account(
        data=RejectServiceAccountRequest(user_id="sa-1", reason="too soon"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )
    assert resp["success"] is True
    # Only the flag is cleared; SA row is NOT deleted.
    mock_prisma.db.litellm_serviceaccounttable.update.assert_called_once()
    update_kwargs = mock_prisma.db.litellm_serviceaccounttable.update.call_args.kwargs
    assert update_kwargs["data"] == {"is_key_rotation_requested": False}
    mock_prisma.db.litellm_serviceaccounttable.delete.assert_not_called()


@pytest.mark.asyncio
async def test_reject_rotation_sends_slack_notification(
    mock_prisma, mock_slack, mocker
):
    """Rotation rejection notifies owners and shows the cleared rotation flag."""
    from unittest.mock import AsyncMock

    mocker.patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test"})
    sa = _make_sa(user_id="sa-1", is_active=True, is_key_rotation_requested=True)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(
        return_value=_NS(user_id="requester-1", user_email="requester@juspay.in")
    )
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(
        return_value=[
            _NS(user_id="owner-1", user_email="owner1@juspay.in"),
            _NS(user_id="owner-2", user_email="owner2@juspay.in"),
        ]
    )

    await sae.reject_service_account(
        data=RejectServiceAccountRequest(user_id="sa-1", reason="too soon"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )

    import asyncio

    await asyncio.sleep(0)
    messages = [
        call.kwargs["message"] for call in mock_slack.send_dm.await_args_list
    ]
    message = next(msg for msg in messages if "rotation rejected" in msg)
    assert "Status: Active" in message
    assert "Key rotation requested: No" in message
    assert "Reason: too soon" in message


@pytest.mark.asyncio
async def test_reject_not_pending_409(mock_prisma, mock_slack):
    from unittest.mock import AsyncMock

    sa = _make_sa(user_id="sa-1", is_active=True, is_key_rotation_requested=False)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    with pytest.raises(Exception):
        await sae.reject_service_account(
            data=RejectServiceAccountRequest(user_id="sa-1"),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
        )


@pytest.mark.asyncio
async def test_reject_not_found_404(mock_prisma, mock_slack):
    from unittest.mock import AsyncMock

    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=None)
    with pytest.raises(Exception):
        await sae.reject_service_account(
            data=RejectServiceAccountRequest(user_id="missing"),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
        )


# ─── GPG public-key validation + encryption ──────────────────────────────────


def test_validate_openpgp_public_key_accepts_armored():
    """A real ASCII-armored public key block passes validation (stripped)."""
    assert sae._validate_openpgp_public_key(_TEST_PUB_KEY) == _TEST_PUB_KEY.strip()


def test_validate_openpgp_public_key_accepts_with_surrounding_whitespace():
    """Leading/trailing whitespace is stripped; the inner armor is what matters."""
    assert sae._validate_openpgp_public_key("  \n" + _TEST_PUB_KEY + "\n  ") == _TEST_PUB_KEY.strip()


def test_validate_openpgp_public_key_rejects_non_armored():
    """A random string without the armor header/footer is rejected with 400."""
    import pytest as _pytest

    with _pytest.raises(Exception) as e:
        sae._validate_openpgp_public_key("just some text, not a key")
    assert getattr(e.value, "status_code", None) == 400


def test_validate_openpgp_public_key_rejects_missing():
    """None / empty are rejected with 400 — a public key is mandatory."""
    import pytest as _pytest

    for bad in (None, "", "   "):
        with _pytest.raises(Exception) as e:
            sae._validate_openpgp_public_key(bad)
        assert getattr(e.value, "status_code", None) == 400


def test_gpg_encrypt_round_trip():
    """_gpg_encrypt produces an armored PGP MESSAGE that decrypts back to the
    plaintext with the matching private key."""
    enc = sae._gpg_encrypt("sk-super-secret-987", _TEST_PUB_KEY)
    assert enc.startswith("-----BEGIN PGP MESSAGE-----")
    assert _decrypt_armored(enc) == "sk-super-secret-987"


@pytest.mark.asyncio
async def test_approve_creation_encrypts_key_when_public_key_present(
    mock_prisma, mock_key_helpers, mock_slack
):
    """Creation-approve with a stored public key returns the GPG-encrypted key
    (not the plaintext) and the encrypted block decrypts back to the minted key."""
    from unittest.mock import AsyncMock

    gen, activate = mock_key_helpers
    sa = _make_sa(
        user_id="sa-1",
        name="my-sa",
        is_active=False,
        is_key_rotation_requested=False,
        public_key=_TEST_PUB_KEY,
        requester="requester-1",
    )
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()

    resp = await sae.approve_service_account(
        data=ApproveServiceAccountRequest(user_id="sa-1", duration="30d", team_id="team-1"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )
    # The plaintext key ("sk-new-key-value" from mock_key_helpers) is encrypted.
    assert resp.key.startswith("-----BEGIN PGP MESSAGE-----")
    assert _decrypt_armored(resp.key) == "sk-new-key-value"


@pytest.mark.asyncio
async def test_approve_creation_fails_when_public_key_missing(
    mock_prisma, mock_key_helpers, mock_slack
):
    """A creation-approve with no public key on file fails with 500 rather than
    leaking the plaintext key back to the caller."""
    from unittest.mock import AsyncMock

    gen, activate = mock_key_helpers
    sa = _make_sa(
        user_id="sa-1",
        name="my-sa",
        is_active=False,
        is_key_rotation_requested=False,
        public_key=None,  # invariant violation
        requester="requester-1",
    )
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()

    with pytest.raises(Exception) as e:
        await sae.approve_service_account(
            data=ApproveServiceAccountRequest(user_id="sa-1", duration="30d", team_id="team-1"),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
        )
    assert getattr(e.value, "status_code", None) == 500
    gen.assert_not_awaited()
    activate.assert_not_awaited()


@pytest.mark.asyncio
async def test_approve_creation_fails_on_encrypt_error(
    mock_prisma, mock_key_helpers, mock_slack
):
    """A malformed public key makes _gpg_encrypt raise 500 — the plaintext key
    is never returned."""
    from unittest.mock import AsyncMock

    _, activate = mock_key_helpers
    sa = _make_sa(
        user_id="sa-1",
        name="my-sa",
        is_active=False,
        is_key_rotation_requested=False,
        # Passes the structural header/footer check but pgpy can't parse it.
        public_key=(
            "-----BEGIN PGP PUBLIC KEY BLOCK-----\n\ngarbage\n-----END PGP PUBLIC KEY BLOCK-----"
        ),
        requester="requester-1",
    )
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    mock_prisma.db.litellm_verificationtoken.update = AsyncMock()

    with pytest.raises(Exception) as e:
        await sae.approve_service_account(
            data=ApproveServiceAccountRequest(user_id="sa-1", duration="30d", team_id="team-1"),
            user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
        )
    assert getattr(e.value, "status_code", None) == 500
    activate.assert_not_awaited()
    mock_prisma.db.litellm_verificationtoken.update.assert_awaited_once_with(
        where={"token": "hashed-new-key"},
        data={"blocked": True},
    )
    mock_prisma.db.litellm_serviceaccounttable.update.assert_awaited_with(
        where={"user_id": "sa-1"},
        data={"is_active": False, "is_key_rotation_requested": False},
    )


def test_gpg_encrypt_reports_missing_python313_imghdr_dependency(mocker):
    """If pgpy's Python 3.13 imghdr compatibility module is missing, report the
    actual missing dependency instead of claiming pgpy itself is absent."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pgpy":
            raise ModuleNotFoundError("No module named 'imghdr'", name="imghdr")
        return real_import(name, *args, **kwargs)

    mocker.patch("builtins.__import__", side_effect=fake_import)

    with pytest.raises(Exception) as e:
        sae._gpg_encrypt("sk-secret", _TEST_PUB_KEY)

    assert getattr(e.value, "status_code", None) == 500
    assert "imghdr" in getattr(e.value, "detail", "")
    assert "standard-imghdr" in getattr(e.value, "detail", "")


@pytest.mark.asyncio
async def test_approve_creation_dm_sent_with_encrypted_key(
    mock_prisma, mock_key_helpers, mock_slack, mocker
):
    """Creation-approve DMs the requester the encrypted key as filename.gpg."""
    from unittest.mock import AsyncMock

    # send_dm only runs when a Slack bot token is configured; without it the
    # helper falls back to the channel webhook. Set one so the DM path runs.
    mocker.patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test"})
    sa = _make_sa(
        user_id="sa-1",
        name="my-sa",
        is_active=False,
        is_key_rotation_requested=False,
        public_key=_TEST_PUB_KEY,
        requester="requester-1",
    )
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    # requester user row resolves to an email so send_dm_file is called.
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(
        return_value=_NS(user_id="requester-1", user_email="requester@juspay.in")
    )

    await sae.approve_service_account(
        data=ApproveServiceAccountRequest(user_id="sa-1", duration="30d", team_id="team-1"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )

    # The DM task is fire-and-forget; let the event loop drain it.
    import asyncio

    await asyncio.sleep(0)
    # send_dm_file was called with the requester's email, decrypt instructions,
    # and the encrypted block as the file content.
    assert mock_slack.send_dm_file.await_count >= 1
    kwargs = mock_slack.send_dm_file.await_args.kwargs
    assert kwargs["user_email"] == "requester@juspay.in"
    assert kwargs["filename"] == "filename.gpg"
    assert kwargs["title"] == "filename.gpg"
    assert kwargs["file_content"].startswith("-----BEGIN PGP MESSAGE-----")
    msg = kwargs["message"]
    assert "Service Account creation approved" in msg
    assert "Requester: requester@juspay.in" in msg
    assert "gpg --decrypt filename.gpg" in msg
    assert "-----BEGIN PGP MESSAGE-----" not in msg


@pytest.mark.asyncio
async def test_approve_creation_encrypted_file_falls_through_to_owner(
    mock_prisma, mock_key_helpers, mock_slack, mocker
):
    """If the requester cannot receive Slack files, try owners before fallback."""
    from unittest.mock import AsyncMock

    mocker.patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test"})
    mock_slack.send_dm_file.side_effect = [False, True]
    sa = _make_sa(
        user_id="sa-1",
        name="my-sa",
        owner_ids=["owner-1"],
        is_active=False,
        is_key_rotation_requested=False,
        public_key=_TEST_PUB_KEY,
        requester="requester-1",
    )
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(
        return_value=_NS(user_id="requester-1", user_email="requester@juspay.in")
    )
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(
        return_value=[_NS(user_id="owner-1", user_email="owner@juspay.in")]
    )

    await sae.approve_service_account(
        data=ApproveServiceAccountRequest(user_id="sa-1", duration="30d", team_id="team-1"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )

    import asyncio

    await asyncio.sleep(0)
    sent_to = [
        call.kwargs["user_email"] for call in mock_slack.send_dm_file.await_args_list
    ]
    assert sent_to == ["requester@juspay.in", "owner@juspay.in"]
    owner_call = mock_slack.send_dm_file.await_args_list[1]
    assert owner_call.kwargs["filename"] == "filename.gpg"
    assert owner_call.kwargs["file_content"].startswith("-----BEGIN PGP MESSAGE-----")
    assert "gpg --decrypt filename.gpg" in owner_call.kwargs["message"]


@pytest.mark.asyncio
async def test_approve_creation_falls_back_to_encrypted_text_when_file_upload_fails(
    mock_prisma, mock_key_helpers, mock_slack, mocker
):
    """If Slack file upload fails, fallback keeps delivery encrypted and visible."""
    from unittest.mock import AsyncMock

    mocker.patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test"})
    mock_slack.send_dm_file.return_value = False
    sa = _make_sa(
        user_id="sa-1",
        name="my-sa",
        is_active=False,
        is_key_rotation_requested=False,
        public_key=_TEST_PUB_KEY,
        requester="requester-1",
    )
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(
        return_value=_NS(user_id="requester-1", user_email="requester@juspay.in")
    )

    await sae.approve_service_account(
        data=ApproveServiceAccountRequest(user_id="sa-1", duration="30d", team_id="team-1"),
        user_api_key_dict=UserAPIKeyAuth(user_id="admin"),
    )

    import asyncio

    await asyncio.sleep(0)
    assert mock_slack.send_dm_file.await_count >= 1
    assert mock_slack.send_dm.await_count >= 1
    messages = [
        call.kwargs["message"] for call in mock_slack.send_dm.await_args_list
    ]
    msg = next(message for message in messages if "Save it as `filename.gpg`" in message)
    assert "Save it as `filename.gpg`" in msg
    assert "gpg --decrypt filename.gpg" in msg
    assert "-----BEGIN PGP MESSAGE-----" in msg


@pytest.mark.asyncio
async def test_build_sa_list_item_includes_public_key_and_requester(mock_prisma, mock_slack):
    """_build_sa_list_item surfaces public_key + requester so the approver UI
    can show a key is on file and who will receive the encrypted key."""
    sa = _make_sa(
        user_id="sa-1",
        public_key=_TEST_PUB_KEY,
        requester="requester-1",
    )
    item = sae._build_sa_list_item(sa, [])
    assert item.public_key == _TEST_PUB_KEY
    assert item.requester == "requester-1"


@pytest.mark.asyncio
async def test_request_rotation_dm_names_requester(mock_prisma, mock_slack, mocker):
    """A rotation request DMs the requester (named in the message) + owners."""
    from unittest.mock import AsyncMock

    mocker.patch.dict(os.environ, {"SLACK_BOT_TOKEN": "xoxb-test"})
    sa = _make_sa(user_id="sa-1", owner_ids=["owner-1", "owner-2"], is_active=True)
    mock_prisma.db.litellm_serviceaccounttable.find_unique = AsyncMock(return_value=sa)
    mock_prisma.db.litellm_serviceaccounttable.update = AsyncMock()
    mock_prisma.db.litellm_usertable.find_unique = AsyncMock(
        return_value=_NS(user_id="owner-1", user_email="owner1@juspay.in")
    )
    mock_prisma.db.litellm_usertable.find_many = AsyncMock(
        return_value=[_NS(user_id="owner-1", user_email="owner1@juspay.in")]
    )

    await sae.request_service_account_rotation(
        data=RequestRotationRequest(user_id="sa-1", requested_by_user_id="owner-1"),
        user_api_key_dict=UserAPIKeyAuth(user_id="owner-1"),
    )
    import asyncio

    await asyncio.sleep(0)
    assert mock_slack.send_dm.await_count >= 1
    msg = mock_slack.send_dm.await_args.kwargs["message"]
    assert "rotation requested" in msg
    assert "owner1@juspay.in" in msg
    assert "Status: Rotation requested" in msg
    assert "Key rotation requested: Yes" in msg
    assert "Use case: batch" in msg
    assert "Requested models: gpt-4" in msg
    assert "Requested RPM limit: 100" in msg
    assert "Requested parallel requests limit: 5" in msg
