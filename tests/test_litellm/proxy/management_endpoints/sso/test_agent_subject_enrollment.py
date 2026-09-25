from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from litellm.proxy.management_endpoints.sso.agent_subject_enrollment import (
    enroll_microsoft_subject,
    microsoft_interactive_subject,
)

TENANT: Final = "11111111-1111-4111-8111-111111111111"
OID: Final = "22222222-2222-4222-8222-222222222222"


def test_enrollment_uses_provider_object_id_and_configured_tenant() -> None:
    subject: Final = microsoft_interactive_subject(
        TENANT, {"id": OID, "mail": "alias@example.com", "tid": "untrusted"}, {}
    )
    assert subject is not None
    assert subject.oid == OID
    assert subject.tenant_id == TENANT
    assert subject.issuer == f"https://login.microsoftonline.com/{TENANT}/v2.0"


@pytest.mark.parametrize("tenant", [None, "common", "organizations", "invalid"])
def test_multitenant_sso_does_not_guess_the_subject_tenant(tenant: str | None) -> None:
    assert microsoft_interactive_subject(tenant, {"id": OID, "tid": TENANT}, {}) is None


@pytest.mark.parametrize("response", [{"mail": "user@example.com"}, {"id": "user@example.com"}, {"id": 42}])
def test_email_and_configurable_aliases_are_not_human_subject_proof(response: dict[str, object]) -> None:
    assert microsoft_interactive_subject(TENANT, response, {}) is None


@pytest.mark.parametrize(
    "endpoint", ["MICROSOFT_USERINFO_ENDPOINT", "MICROSOFT_TOKEN_ENDPOINT", "MICROSOFT_AUTHORIZATION_ENDPOINT"]
)
def test_custom_provider_endpoints_do_not_enroll_trusted_microsoft_subjects(endpoint: str) -> None:
    assert microsoft_interactive_subject(TENANT, {"id": OID}, {endpoint: "https://custom.example"}) is None


@pytest.mark.asyncio
async def test_interactive_enrollment_preserves_the_canonical_local_user() -> None:
    table: Final = AsyncMock()
    table.upsert.return_value = SimpleNamespace(user_id="canonical", verified_via="sso_interactive")
    client: Final = SimpleNamespace(db=SimpleNamespace(litellm_verifiedhumansubject=table))
    subject: Final = microsoft_interactive_subject(TENANT, {"id": OID}, {})
    assert subject is not None
    await enroll_microsoft_subject(subject, "canonical", client)
    table.upsert.assert_awaited_once_with(
        where={"issuer_tenant_id_oid": {"issuer": subject.issuer, "tenant_id": TENANT, "oid": OID}},
        data={
            "create": {
                "issuer": subject.issuer,
                "tenant_id": TENANT,
                "oid": OID,
                "user_id": "canonical",
                "verified_via": "sso_interactive",
            },
            "update": {},
        },
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id,verified_via", [("another-user", "sso_interactive"), ("canonical", "untrusted")])
async def test_interactive_enrollment_does_not_reassign_an_existing_subject(user_id: str, verified_via: str) -> None:
    table: Final = AsyncMock()
    table.upsert.return_value = SimpleNamespace(user_id=user_id, verified_via=verified_via)
    client: Final = SimpleNamespace(db=SimpleNamespace(litellm_verifiedhumansubject=table))
    with pytest.raises(HTTPException) as failure:
        await enroll_microsoft_subject(microsoft_interactive_subject(TENANT, {"id": OID}, {}), "canonical", client)
    assert failure.value.status_code == 403
    assert table.upsert.call_args.kwargs["data"]["update"] == {}


@pytest.mark.asyncio
async def test_enrollment_storage_failure_is_not_a_successful_login() -> None:
    table: Final = AsyncMock()
    table.upsert.side_effect = RuntimeError("database unavailable")
    client: Final = SimpleNamespace(db=SimpleNamespace(litellm_verifiedhumansubject=table))
    with pytest.raises(HTTPException) as failure:
        await enroll_microsoft_subject(microsoft_interactive_subject(TENANT, {"id": OID}, {}), "canonical", client)
    assert failure.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", [None, "", 42])
async def test_enrollment_requires_a_canonical_local_user(user_id: object) -> None:
    table: Final = AsyncMock()
    client: Final = SimpleNamespace(db=SimpleNamespace(litellm_verifiedhumansubject=table))
    await enroll_microsoft_subject(microsoft_interactive_subject(TENANT, {"id": OID}, {}), user_id, client)
    table.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_untrusted_metadata_cannot_enroll_a_human() -> None:
    table: Final = AsyncMock()
    client: Final = SimpleNamespace(db=SimpleNamespace(litellm_verifiedhumansubject=table))
    await enroll_microsoft_subject({"issuer": "forged", "tenant_id": TENANT, "oid": OID}, "canonical", client)
    table.upsert.assert_not_awaited()
