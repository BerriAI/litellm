import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

import httpx
import pytest
from integration._support import codex_vendor as cv
from integration._support import responses_vendor as rv
from integration._support.client import Gateway, Scenario, eventually, gateway_from_environment, string_value
from integration._support.process import OwnedProxy, owned_proxy_process
from integration._support.wire import Wire, wire_server
from pydantic import BaseModel, ConfigDict, JsonValue, TypeAdapter

_MODEL: Final = "chatgpt/gpt-5.5"
_PROVIDER: Final = "chatgpt"
_FEE_TABLE_QUERY: Final = (
    'SELECT subscription_account_id FROM "LiteLLM_SubscriptionAccountTable" WHERE custom_llm_provider = %s'
    " AND account_id = %s"
)
_FEE_ROW_QUERY: Final = (
    'SELECT subscription_account_id FROM "LiteLLM_SubscriptionAccountTable" WHERE subscription_account_id = %s'
)
_MONTHLY_FEE: Final = 200.0
_BILLING_PERIOD_START: Final = "2026-01-15"
_RANGE: Final = {"start_date": "2026-03-01", "end_date": "2026-03-31"}
_PERIOD_IN_RANGE: Final = ("2026-03-15", "2026-04-14")
_TOKENS: Final = TypeAdapter(int)


class _Metrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    spend: float
    prompt_tokens: int
    completion_tokens: int
    api_requests: int
    successful_requests: int
    failed_requests: int


class _ProviderUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    metrics: _Metrics
    metadata: Mapping[str, JsonValue]


class _Breakdown(BaseModel):
    model_config = ConfigDict(frozen=True)

    providers: Mapping[str, _ProviderUsage]


class _Day(BaseModel):
    model_config = ConfigDict(frozen=True)

    breakdown: _Breakdown


class _DailyActivity(BaseModel):
    model_config = ConfigDict(frozen=True)

    results: tuple[_Day, ...]


class _Fee(BaseModel):
    model_config = ConfigDict(frozen=True)

    subscription_account_id: str
    custom_llm_provider: str
    account_id: str
    monthly_fee: float
    currency: str
    billing_period_start: str


class _Period(BaseModel):
    model_config = ConfigDict(frozen=True)

    start: str
    end: str


class _Account(BaseModel):
    model_config = ConfigDict(frozen=True)

    custom_llm_provider: str
    account_id: str | None
    deployments: tuple[str, ...]
    fee: _Fee | None
    billing_periods: tuple[_Period, ...]
    fixed_cost: float | None


class _SubscriptionUsage(BaseModel):
    model_config = ConfigDict(frozen=True)

    accounts: tuple[_Account, ...]


@dataclass(frozen=True, slots=True)
class _Rig:
    wire: Wire
    proxy: OwnedProxy
    model: str

    @property
    def gateway(self) -> Gateway:
        return self.proxy.gateway


@pytest.fixture(scope="module")
def rig(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Rig]:
    directory: Final = tmp_path_factory.mktemp("chatgpt-subscription-usage")
    no_default_login: Final = directory / "no-default-login"
    no_default_login.mkdir()
    deployment_login: Final = cv.login(directory) / "auth.json"
    with gateway_from_environment() as gateway, wire_server(cv.CodexVendor().respond) as wire:
        overrides: Final = {"CHATGPT_TOKEN_DIR": str(no_default_login), "CHATGPT_API_BASE": wire.url}
        config: Final = cv.proxy_config(directory, probe=False)
        with (
            owned_proxy_process(gateway, directory, overrides, config=config, workers=2) as owned,
            owned.gateway.scenario() as scenario,
        ):
            model: Final = scenario.model(
                model=_MODEL, api_base=wire.url, api_key=None, chatgpt_auth_file=str(deployment_login)
            )
            yield _Rig(wire, owned, model)


def _responses_body(model: str, marker: str, *, stream: bool) -> Mapping[str, JsonValue]:
    return {
        "model": model,
        "input": [{"role": "user", "content": f"Say marker-{marker}"}],
        "stream": stream,
        "cache": {"no-cache": True},
    }


def _assert_answered(response: httpx.Response, marker: str, content_type: str) -> None:
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith(content_type), response.headers
    assert rv.answer(marker) in response.text, response.text


def _today() -> str:
    return datetime.now(UTC).date().isoformat()


def _chatgpt_usage_between(gateway: Gateway, first_day: str, last_day: str) -> tuple[_ProviderUsage, ...]:
    response: Final = gateway.request(
        "GET", "/user/daily/activity", params={"start_date": first_day, "end_date": last_day}
    )
    assert response.status_code == 200, response.text
    activity: Final = _DailyActivity.model_validate_json(response.content)
    return tuple(
        result.breakdown.providers[_PROVIDER] for result in activity.results if _PROVIDER in result.breakdown.providers
    )


def _summed(buckets: tuple[_ProviderUsage, ...]) -> _Metrics:
    return _Metrics(
        spend=sum(bucket.metrics.spend for bucket in buckets),
        prompt_tokens=sum(bucket.metrics.prompt_tokens for bucket in buckets),
        completion_tokens=sum(bucket.metrics.completion_tokens for bucket in buckets),
        api_requests=sum(bucket.metrics.api_requests for bucket in buckets),
        successful_requests=sum(bucket.metrics.successful_requests for bucket in buckets),
        failed_requests=sum(bucket.metrics.failed_requests for bucket in buckets),
    )


def _usage(gateway: Gateway, *, key: str | None = None) -> httpx.Response:
    return gateway.request("GET", "/subscription_accounts/usage", params=_RANGE, key=key)


def _accounts(gateway: Gateway) -> tuple[_Account, ...]:
    response: Final = _usage(gateway)
    assert response.status_code == 200, response.text
    return _SubscriptionUsage.model_validate_json(response.content).accounts


def _deployment_account(rig: _Rig) -> _Account:
    (account,) = _accounts(rig.gateway)
    assert (account.custom_llm_provider, account.account_id, account.deployments) == (
        _PROVIDER,
        cv.ACCOUNT,
        (rig.model,),
    ), account
    return account


def _assert_no_fee(account: _Account) -> None:
    assert (account.fee, account.billing_periods, account.fixed_cost) == (None, (), None), account


def _fee_body(account_id: str, monthly_fee: float = _MONTHLY_FEE) -> Mapping[str, JsonValue]:
    return {
        "custom_llm_provider": _PROVIDER,
        "account_id": account_id,
        "monthly_fee": monthly_fee,
        "currency": "USD",
        "billing_period_start": _BILLING_PERIOD_START,
        "label": "integration",
    }


def _delete_fee_if_present(gateway: Gateway, identity: str) -> None:
    if gateway.rows(_FEE_ROW_QUERY, (identity,)):
        gateway.post("/subscription_accounts/delete", {"subscription_account_id": identity})
    assert gateway.rows(_FEE_ROW_QUERY, (identity,)) == []


def _set_fee(scenario: Scenario, account_id: str) -> str:
    created: Final = scenario.gateway.post("/subscription_accounts/new", _fee_body(account_id))
    identity: Final = string_value(created["subscription_account_id"])
    scenario.cleanups.callback(_delete_fee_if_present, scenario.gateway, identity)
    return identity


def _assert_fee_reported(account: _Account, identity: str) -> None:
    assert account.fee is not None, account
    assert (
        account.fee.subscription_account_id,
        account.fee.account_id,
        account.fee.monthly_fee,
        account.fee.currency,
        account.fee.billing_period_start,
    ) == (identity, cv.ACCOUNT, _MONTHLY_FEE, "USD", _BILLING_PERIOD_START), account
    assert tuple((period.start, period.end) for period in account.billing_periods) == (_PERIOD_IN_RANGE,), account
    assert account.fixed_cost == _MONTHLY_FEE, account


def test_responses_calls_on_a_deployment_login_count_as_subscription_covered_chatgpt_usage(rig: _Rig) -> None:
    first_day: Final = _today()
    rig.wire.drain()
    plain_marker: Final = uuid.uuid4().hex
    stream_marker: Final = uuid.uuid4().hex
    plain: Final = rig.gateway.request("POST", "/v1/responses", _responses_body(rig.model, plain_marker, stream=False))
    _assert_answered(plain, plain_marker, "application/json")
    streamed: Final = rig.gateway.request(
        "POST", "/v1/responses", _responses_body(rig.model, stream_marker, stream=True)
    )
    _assert_answered(streamed, stream_marker, "text/event-stream")
    forwarded: Final = rig.wire.drain()
    assert len(forwarded) == 2, [request.target for request in forwarded]
    for request, marker in zip(forwarded, (plain_marker, stream_marker), strict=True):
        cv.forwarded(request, marker)
    last_day: Final = _today()
    buckets: Final = eventually(
        lambda: _chatgpt_usage_between(rig.gateway, first_day, last_day),
        lambda found: _summed(found).api_requests >= 2,
        seconds=70,
    )
    totals: Final = _summed(buckets)
    assert (totals.api_requests, totals.successful_requests, totals.failed_requests, totals.spend) == (
        2,
        2,
        0,
        0,
    ), buckets
    assert (totals.prompt_tokens, totals.completion_tokens) == (
        2 * _TOKENS.validate_python(cv.USAGE["input_tokens"]),
        2 * _TOKENS.validate_python(cv.USAGE["output_tokens"]),
    ), buckets
    assert all(bucket.metadata.get("subscription_covered") is True for bucket in buckets), buckets


def test_fee_on_the_deployment_account_reports_its_fixed_cost_until_deleted(rig: _Rig) -> None:
    _assert_no_fee(_deployment_account(rig))
    with rig.gateway.scenario() as scenario:
        identity: Final = _set_fee(scenario, cv.ACCOUNT)
        _assert_fee_reported(_deployment_account(rig), identity)
        deleted: Final = rig.gateway.post("/subscription_accounts/delete", {"subscription_account_id": identity})
        assert deleted["subscription_account_id"] == identity, deleted
        assert rig.gateway.rows(_FEE_ROW_QUERY, (identity,)) == []
        _assert_no_fee(_deployment_account(rig))


def test_second_fee_for_the_same_account_answers_400_and_keeps_the_first(rig: _Rig) -> None:
    with rig.gateway.scenario() as scenario:
        identity: Final = _set_fee(scenario, cv.ACCOUNT)
        duplicate: Final = rig.gateway.request(
            "POST", "/subscription_accounts/new", _fee_body(cv.ACCOUNT, monthly_fee=3 * _MONTHLY_FEE)
        )
        assert duplicate.status_code == 400, duplicate.text
        assert rig.gateway.rows(_FEE_TABLE_QUERY, (_PROVIDER, cv.ACCOUNT)) == [{"subscription_account_id": identity}]
        _assert_fee_reported(_deployment_account(rig), identity)


def test_update_and_delete_of_an_unknown_fee_answer_404(rig: _Rig) -> None:
    unknown: Final = str(uuid.uuid4())
    updated: Final = rig.gateway.request(
        "POST", "/subscription_accounts/update", {"subscription_account_id": unknown, "monthly_fee": _MONTHLY_FEE}
    )
    assert updated.status_code == 404, updated.text
    deleted: Final = rig.gateway.request("POST", "/subscription_accounts/delete", {"subscription_account_id": unknown})
    assert deleted.status_code == 404, deleted.text
    assert rig.gateway.rows(_FEE_ROW_QUERY, (unknown,)) == []


def test_internal_user_is_refused_setting_a_fee_and_reading_usage(rig: _Rig) -> None:
    with rig.gateway.scenario() as scenario:
        key: Final = scenario.key(user_id=scenario.user(user_role="internal_user"))
        created: Final = rig.gateway.request("POST", "/subscription_accounts/new", _fee_body(cv.ACCOUNT), key=key)
        assert created.status_code == 401, created.text
        assert rig.gateway.rows(_FEE_TABLE_QUERY, (_PROVIDER, cv.ACCOUNT)) == []
        usage: Final = _usage(rig.gateway, key=key)
        assert usage.status_code == 401, usage.text


def test_fee_on_an_account_no_deployment_uses_stays_listed_and_counted(rig: _Rig) -> None:
    unused: Final = f"acct-unused-{uuid.uuid4().hex}"
    with rig.gateway.scenario() as scenario:
        identity: Final = _set_fee(scenario, unused)
        accounts: Final = _accounts(rig.gateway)
        assert frozenset(account.account_id for account in accounts) == {cv.ACCOUNT, unused}, accounts
        (unused_account,) = (account for account in accounts if account.account_id == unused)
        assert unused_account.fee is not None, unused_account
        assert (unused_account.deployments, unused_account.fee.subscription_account_id) == ((), identity)
        assert unused_account.fixed_cost == _MONTHLY_FEE, unused_account
