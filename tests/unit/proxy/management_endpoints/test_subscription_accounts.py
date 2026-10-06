from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Final
from uuid import uuid4

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from litellm.models.subscription_account import LiteLLM_SubscriptionAccountTable
from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.proxy.management_endpoints.subscription_accounts import (
    SubscriptionDeployment,
    get_subscription_account_repository,
    get_subscription_deployments,
    router,
    subscription_deployments_from_model_list,
)
from litellm.repositories.subscription_account_repository import (
    DuplicateSubscriptionAccount,
    SubscriptionAccountFeeChanges,
    SubscriptionAccountFeeFields,
)

_ACCOUNT: Final = "org-team-workspace"
_TEST_FEE: Final = {
    "custom_llm_provider": "chatgpt",
    "account_id": _ACCOUNT,
    "monthly_fee": 25.0,
    "currency": "USD",
    "billing_period_start": "2026-10-01",
    "label": "Team plan (test value)",
}
_TWO_DEPLOYMENTS_ONE_ACCOUNT: Final = (
    SubscriptionDeployment(model_name="gpt-5.6-terra", custom_llm_provider="chatgpt", account_id=_ACCOUNT),
    SubscriptionDeployment(model_name="gpt-5.6-sol", custom_llm_provider="chatgpt", account_id=_ACCOUNT),
)


class _FakeStore:
    def __init__(self) -> None:
        self.rows: dict[str, LiteLLM_SubscriptionAccountTable] = {}
        self.written_columns: list[dict[str, float | str | None]] = []

    async def list_all(self) -> list[LiteLLM_SubscriptionAccountTable]:
        return sorted(self.rows.values(), key=lambda row: row.created_at)

    async def find_by_id(self, id_value: str) -> LiteLLM_SubscriptionAccountTable | None:
        return self.rows.get(id_value)

    async def create_account(
        self, custom_llm_provider: str, account_id: str, fee: SubscriptionAccountFeeFields, created_by: str
    ) -> LiteLLM_SubscriptionAccountTable | DuplicateSubscriptionAccount:
        for row in self.rows.values():
            if (row.custom_llm_provider, row.account_id) == (custom_llm_provider, account_id):
                return DuplicateSubscriptionAccount(custom_llm_provider=custom_llm_provider, account_id=account_id)
        now = datetime.now(timezone.utc)
        row = LiteLLM_SubscriptionAccountTable(
            subscription_account_id=str(uuid4()),
            custom_llm_provider=custom_llm_provider,
            account_id=account_id,
            label=fee.label,
            monthly_fee=fee.monthly_fee,
            currency=fee.currency,
            billing_period_start=fee.billing_period_start,
            created_at=now,
            created_by=created_by,
            updated_at=now,
            updated_by=created_by,
        )
        self.rows[row.subscription_account_id] = row
        return row

    async def update_fee(
        self, subscription_account_id: str, changes: SubscriptionAccountFeeChanges, updated_by: str
    ) -> LiteLLM_SubscriptionAccountTable | None:
        columns = dict(changes.columns())
        self.written_columns.append(columns)
        existing = self.rows.get(subscription_account_id)
        if existing is None:
            return None
        updated = existing.model_copy(
            update={**columns, "updated_by": updated_by, "updated_at": datetime.now(timezone.utc)}
        )
        self.rows[subscription_account_id] = updated
        return updated

    async def delete_account(self, subscription_account_id: str) -> LiteLLM_SubscriptionAccountTable | None:
        return self.rows.pop(subscription_account_id, None)


def _resolve_auth(request: Request) -> UserAPIKeyAuth:
    role: Final = LitellmUserRoles(request.headers.get("x-user-role", LitellmUserRoles.PROXY_ADMIN.value))
    return UserAPIKeyAuth(user_id="admin", user_role=role, api_key="sk-test")


@pytest.fixture
def deployments() -> tuple[SubscriptionDeployment, ...]:
    return _TWO_DEPLOYMENTS_ONE_ACCOUNT


@pytest.fixture
def client(deployments: tuple[SubscriptionDeployment, ...]) -> Iterator[tuple[TestClient, _FakeStore]]:
    store: Final = _FakeStore()
    app: Final = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_subscription_account_repository] = lambda: store
    app.dependency_overrides[get_subscription_deployments] = lambda: deployments
    app.dependency_overrides[user_api_key_auth] = _resolve_auth
    with TestClient(app) as test_client:
        yield test_client, store


def _usage(test_client: TestClient, start_date: str, end_date: str, **headers: str) -> dict:
    response = test_client.get(
        "/subscription_accounts/usage", params={"start_date": start_date, "end_date": end_date}, headers=headers
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_untracked_account_lists_its_deployments_with_no_fee(client):
    test_client, _ = client

    body = _usage(test_client, "2026-10-01", "2026-10-07")

    assert body["subscription_providers"] == ["chatgpt"]
    assert body["accounts"] == [
        {
            "custom_llm_provider": "chatgpt",
            "account_id": _ACCOUNT,
            "deployments": ["gpt-5.6-sol", "gpt-5.6-terra"],
            "fee": None,
            "billing_periods": [],
            "fixed_cost": None,
        }
    ]


def test_fee_counts_once_per_billing_period_however_many_deployments_share_the_account(client):
    test_client, _ = client
    created = test_client.post("/subscription_accounts/new", json=_TEST_FEE)
    assert created.status_code == 200, created.text

    one_period = _usage(test_client, "2026-10-01", "2026-10-07")
    two_periods = _usage(test_client, "2026-09-15", "2026-11-15")
    between_starts = _usage(test_client, "2026-10-02", "2026-10-31")

    (account,) = one_period["accounts"]
    assert account["deployments"] == ["gpt-5.6-sol", "gpt-5.6-terra"]
    assert account["fee"]["monthly_fee"] == 25.0
    assert account["fee"]["currency"] == "USD"
    assert account["fee"]["label"] == "Team plan (test value)"
    assert account["billing_periods"] == [{"start": "2026-10-01", "end": "2026-10-31"}]
    assert account["fixed_cost"] == 25.0
    assert two_periods["accounts"][0]["fixed_cost"] == 50.0
    assert [period["start"] for period in two_periods["accounts"][0]["billing_periods"]] == [
        "2026-10-01",
        "2026-11-01",
    ]
    assert between_starts["accounts"][0]["fixed_cost"] == 0.0
    assert between_starts["accounts"][0]["billing_periods"] == []


def test_fee_without_a_matching_deployment_is_still_listed(client):
    test_client, _ = client
    other = {**_TEST_FEE, "account_id": "org-other", "monthly_fee": 200.0, "currency": "EUR"}
    assert test_client.post("/subscription_accounts/new", json=other).status_code == 200

    body = _usage(test_client, "2026-10-01", "2026-10-07")

    assert [account["account_id"] for account in body["accounts"]] == ["org-other", _ACCOUNT]
    assert body["accounts"][0]["deployments"] == []
    assert body["accounts"][0]["fee"]["currency"] == "EUR"
    assert body["accounts"][0]["fixed_cost"] == 200.0
    assert body["accounts"][1]["fee"] is None


@pytest.mark.parametrize(
    "deployments",
    [(SubscriptionDeployment(model_name="gpt-5.6-terra", custom_llm_provider="chatgpt", account_id=None),)],
)
def test_account_not_signed_in_shows_no_account_and_takes_no_fee(client):
    test_client, _ = client
    assert test_client.post("/subscription_accounts/new", json=_TEST_FEE).status_code == 200

    body = _usage(test_client, "2026-10-01", "2026-10-07")

    unsigned = next(account for account in body["accounts"] if account["account_id"] is None)
    assert unsigned["deployments"] == ["gpt-5.6-terra"]
    assert unsigned["fee"] is None
    assert unsigned["fixed_cost"] is None


def test_update_changes_the_fee_and_delete_returns_the_account_to_untracked(client):
    test_client, store = client
    created = test_client.post("/subscription_accounts/new", json=_TEST_FEE).json()
    account_id = created["subscription_account_id"]

    updated = test_client.post(
        "/subscription_accounts/update",
        json={"subscription_account_id": account_id, "monthly_fee": 30.0, "billing_period_start": "2026-10-05"},
    )
    assert updated.status_code == 200, updated.text
    assert updated.json()["monthly_fee"] == 30.0
    assert updated.json()["currency"] == "USD"
    assert updated.json()["billing_period_start"] == "2026-10-05"
    assert updated.json()["label"] == _TEST_FEE["label"]
    assert _usage(test_client, "2026-10-01", "2026-10-07")["accounts"][0]["fixed_cost"] == 30.0

    cleared = test_client.post(
        "/subscription_accounts/update", json={"subscription_account_id": account_id, "label": None}
    )
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["label"] is None
    assert cleared.json()["monthly_fee"] == 30.0
    assert store.written_columns == [{"monthly_fee": 30.0, "billing_period_start": "2026-10-05"}, {"label": None}]

    unknown = test_client.post(
        "/subscription_accounts/update", json={"subscription_account_id": "no-such-account", "monthly_fee": 1.0}
    )
    assert unknown.status_code == 404

    deleted = test_client.post("/subscription_accounts/delete", json={"subscription_account_id": account_id})
    assert deleted.status_code == 200, deleted.text
    assert store.rows == {}
    assert _usage(test_client, "2026-10-01", "2026-10-07")["accounts"][0]["fixed_cost"] is None

    missing = test_client.post("/subscription_accounts/delete", json={"subscription_account_id": account_id})
    assert missing.status_code == 404


def test_rejects_a_second_fee_for_the_same_account(client):
    test_client, _ = client
    assert test_client.post("/subscription_accounts/new", json=_TEST_FEE).status_code == 200

    duplicate = test_client.post("/subscription_accounts/new", json={**_TEST_FEE, "monthly_fee": 99.0})

    assert duplicate.status_code == 400
    assert "/subscription_accounts/update" in duplicate.json()["detail"]["error"]


def test_rejects_a_fee_on_a_metered_provider(client):
    test_client, store = client

    response = test_client.post("/subscription_accounts/new", json={**_TEST_FEE, "custom_llm_provider": "openai"})

    assert response.status_code == 400
    assert store.rows == {}


@pytest.mark.parametrize(
    "payload",
    [
        {**_TEST_FEE, "monthly_fee": 0},
        {**_TEST_FEE, "currency": "usd"},
        {**_TEST_FEE, "currency": "$"},
        {**_TEST_FEE, "billing_period_start": "10/01/2026"},
    ],
)
def test_rejects_malformed_fee_fields(client, payload):
    test_client, store = client

    assert test_client.post("/subscription_accounts/new", json=payload).status_code == 422
    assert store.rows == {}


def test_usage_rejects_a_malformed_or_inverted_range(client):
    test_client, _ = client

    malformed = [
        test_client.get("/subscription_accounts/usage", params={"start_date": start, "end_date": "2026-10-07"})
        for start in ("2026-10-1", "20261001", "2026-W40-1", "2026-13-01")
    ]
    inverted = test_client.get(
        "/subscription_accounts/usage", params={"start_date": "2026-10-07", "end_date": "2026-10-01"}
    )

    assert [response.status_code for response in malformed] == [400, 400, 400, 400]
    assert inverted.status_code == 400


def test_viewer_reads_usage_but_cannot_write(client):
    test_client, store = client
    viewer = {"x-user-role": LitellmUserRoles.PROXY_ADMIN_VIEW_ONLY.value}

    assert _usage(test_client, "2026-10-01", "2026-10-07", **viewer)["accounts"][0]["account_id"] == _ACCOUNT
    assert test_client.post("/subscription_accounts/new", json=_TEST_FEE, headers=viewer).status_code == 403
    assert store.rows == {}


def test_internal_user_cannot_read_subscription_accounts(client):
    test_client, _ = client

    response = test_client.get(
        "/subscription_accounts/usage",
        params={"start_date": "2026-10-01", "end_date": "2026-10-07"},
        headers={"x-user-role": LitellmUserRoles.INTERNAL_USER.value},
    )

    assert response.status_code == 403


def test_deployments_resolve_by_provider_and_share_one_account_lookup():
    model_list = [
        {"model_name": "gpt-5.6-terra", "litellm_params": {"model": "chatgpt/gpt-5.6-terra"}},
        {"model_name": "gpt-5.6-sol", "litellm_params": {"model": "chatgpt/gpt-5.6-sol"}},
        {"model_name": "sonnet", "litellm_params": {"model": "anthropic/claude-sonnet-4-5", "api_key": "sk-x"}},
        {"model_name": "broken", "litellm_params": {"model": "no-such-provider-model"}},
    ]
    lookups: list[str] = []

    def resolver(provider: str) -> str | None:
        lookups.append(provider)
        return _ACCOUNT

    deployments = subscription_deployments_from_model_list(model_list, resolver)

    assert deployments == _TWO_DEPLOYMENTS_ONE_ACCOUNT
    assert lookups == ["chatgpt"]
