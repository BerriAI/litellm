import asyncio
from typing import Final

import pytest
import pytest_asyncio
from pydantic import ValidationError

from litellm.proxy.management_endpoints.liteask.approval import (
    ApprovalError,
    SealedApproval,
    claim_approval,
    open_approval,
    seal_approval,
)
from litellm.proxy.management_endpoints.liteask.models import LiteAskApprovalRequest
from tests.test_litellm.proxy.management_endpoints.liteask.conftest import AtomicApprovalStore


@pytest_asyncio.fixture
async def proposal(monkeypatch: pytest.MonkeyPatch, approval_store: AtomicApprovalStore) -> tuple[str, int]:
    monkeypatch.setenv("LITELLM_SALT_KEY", "liteask-approval-test-only")
    sealed: Final = await seal_approval(
        user_id="admin",
        credential="credential-fingerprint",
        conversation_id="conversation",
        tool="team_update",
        arguments={"body": {"team_id": "team", "max_budget": 100}},
        now=1000,
        store=approval_store,
    )
    assert not isinstance(sealed, ApprovalError)
    return sealed


@pytest.mark.parametrize(
    ("user_id", "credential", "conversation_id", "now"),
    (
        ("other", "credential-fingerprint", "conversation", 1001),
        ("admin", "other", "conversation", 1001),
        ("admin", "credential-fingerprint", "other", 1001),
        ("admin", "credential-fingerprint", "conversation", 1300),
    ),
)
def test_approval_is_bound_to_actor_credential_conversation_and_expiry(
    proposal: tuple[str, int],
    user_id: str,
    credential: str,
    conversation_id: str,
    now: int,
) -> None:
    result: Final = open_approval(
        proposal[0],
        user_id=user_id,
        credential=credential,
        conversation_id=conversation_id,
        now=now,
    )
    assert isinstance(result, ApprovalError)


def test_sealed_arguments_cannot_be_tampered_or_replaced(proposal: tuple[str, int]) -> None:
    payload: Final = open_approval(
        proposal[0],
        user_id="admin",
        credential="credential-fingerprint",
        conversation_id="conversation",
        now=1001,
    )
    assert isinstance(payload, SealedApproval)
    assert payload.arguments == {"body": {"team_id": "team", "max_budget": 100}}
    tampered: Final = open_approval(
        proposal[0][:-8] + "XXXXXXXX",
        user_id="admin",
        credential="credential-fingerprint",
        conversation_id="conversation",
        now=1001,
    )
    assert isinstance(tampered, ApprovalError)
    with pytest.raises(ValidationError):
        LiteAskApprovalRequest.model_validate(
            {
                "conversation_id": "ce52e0ef-e86e-4a7b-a884-1760a452d428",
                "token": proposal[0],
                "arguments": {"body": {"max_budget": 999}},
            }
        )


@pytest.mark.asyncio
async def test_concurrent_claims_have_one_winner_and_storage_failure_never_allows_write(
    proposal: tuple[str, int], approval_store: AtomicApprovalStore
) -> None:
    payload: Final = open_approval(
        proposal[0],
        user_id="admin",
        credential="credential-fingerprint",
        conversation_id="conversation",
        now=1001,
    )
    assert isinstance(payload, SealedApproval)
    results: Final = await asyncio.gather(*(claim_approval(payload, store=approval_store, now=1001) for _ in range(8)))
    assert sum(result is None for result in results) == 1
    assert all(result is None or result.status_code == 409 for result in results)
    unavailable: Final = await claim_approval(payload, store=AtomicApprovalStore(unavailable=True), now=1001)
    missing: Final = await claim_approval(payload, store=None, now=1001)
    assert isinstance(unavailable, ApprovalError) and unavailable.status_code == 503
    assert isinstance(missing, ApprovalError) and missing.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("lose_before_first_claim", (True, False))
async def test_lost_or_unissued_nonce_never_becomes_an_executable_approval(
    proposal: tuple[str, int], approval_store: AtomicApprovalStore, lose_before_first_claim: bool
) -> None:
    payload: Final = open_approval(
        proposal[0],
        user_id="admin",
        credential="credential-fingerprint",
        conversation_id="conversation",
        now=1001,
    )
    assert isinstance(payload, SealedApproval)
    if not lose_before_first_claim:
        assert await claim_approval(payload, store=approval_store, now=1001) is None
    approval_store.lose_state()
    lost: Final = await claim_approval(payload, store=approval_store, now=1001)
    unissued: Final = await claim_approval(payload, store=AtomicApprovalStore(), now=1001)
    assert isinstance(lost, ApprovalError) and lost.status_code == 409
    assert isinstance(unissued, ApprovalError) and unissued.status_code == 409


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "store",
    (
        None,
        AtomicApprovalStore(unavailable=True),
        AtomicApprovalStore(issue_result=None),
        AtomicApprovalStore(issue_result=0),
        AtomicApprovalStore(issue_result=True),
    ),
)
async def test_unacknowledged_issuance_never_returns_a_token(
    monkeypatch: pytest.MonkeyPatch, store: AtomicApprovalStore | None
) -> None:
    monkeypatch.setenv("LITELLM_SALT_KEY", "liteask-approval-test-only")
    result: Final = await seal_approval(
        user_id="admin",
        credential="credential-fingerprint",
        conversation_id="conversation",
        tool="team_update",
        arguments={"body": {"team_id": "team", "max_budget": 100}},
        now=1000,
        store=store,
    )
    assert isinstance(result, ApprovalError) and result.status_code == 503
