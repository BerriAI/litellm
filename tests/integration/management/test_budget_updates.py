from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

from tests.integration._support.client import Gateway, string_value
from tests.integration._support.database import read_rows
from tests.integration._support.process import owned_proxy


def _persisted_reset_at(budget_id: str) -> datetime:
    rows: Final = read_rows(
        'SELECT budget_reset_at::text AS reset_at FROM "LiteLLM_BudgetTable" WHERE budget_id = %s', (budget_id,)
    )
    assert len(rows) == 1, rows
    reset_at: Final = datetime.fromisoformat(string_value(rows[0]["reset_at"]))
    return reset_at if reset_at.tzinfo is not None else reset_at.replace(tzinfo=timezone.utc)


@pytest.mark.covers("mgmt.budget.update.duration_change_recomputes_reset_at")
def test_shortening_budget_duration_moves_reset_at_onto_the_new_schedule(gateway: Gateway) -> None:
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(max_budget=10.0, budget_duration="10d")
        ten_day_reset_at: Final = _persisted_reset_at(budget_id)
        before: Final = datetime.now(timezone.utc)
        response: Final = gateway.request("POST", "/budget/update", {"budget_id": budget_id, "budget_duration": "1d"})
        assert response.status_code == 200, response.text
        updated: Final = _persisted_reset_at(budget_id)
        assert updated < ten_day_reset_at, f"{updated} not before {ten_day_reset_at}"
        assert before < updated <= before + timedelta(days=1, minutes=5), f"{updated} not within 1d of {before}"


BUDGET_COLUMNS: Final = (
    "budget_id, max_budget, soft_budget, max_parallel_requests, tpm_limit, rpm_limit, tpd_limit, model_max_budget, "
    "budget_duration, allowed_models, temp_budget_increase, temp_budget_expiry, created_by, updated_by"
)
VOLATILE_INFO_FIELDS: Final = frozenset({"created_at", "updated_at", "budget_reset_at"})
UNLOADED_RELATIONS: Final[dict[str, JsonValue]] = dict.fromkeys(
    (
        "organization",
        "projects",
        "keys",
        "end_users",
        "tags",
        "model_access_groups",
        "team_membership",
        "organization_membership",
    )
)


def _budget_row(budget_id: str) -> dict[str, JsonValue]:
    rows: Final = read_rows(
        f'SELECT {BUDGET_COLUMNS}, budget_reset_at::text AS reset_at FROM "LiteLLM_BudgetTable" WHERE budget_id = %s',
        (budget_id,),
    )
    assert len(rows) == 1, rows
    return rows[0]


def _budget_info(gateway: Gateway, budget_id: str) -> dict[str, JsonValue]:
    response: Final = gateway.request("POST", "/budget/info", {"budgets": [budget_id]})
    assert response.status_code == 200, response.text
    entries: Final = TypeAdapter(list[dict[str, JsonValue]]).validate_json(response.content)
    assert len(entries) == 1, entries
    return entries[0]


def _reset_at(value: JsonValue) -> datetime | None:
    if value is None:
        return None
    parsed: Final = datetime.fromisoformat(string_value(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=timezone.utc)


def _assert_budget(
    gateway: Gateway,
    budget_id: str,
    expected: dict[str, JsonValue],
    reset_after: datetime | None,
    reset_within: timedelta,
) -> None:
    row: Final = _budget_row(budget_id)
    reset_at: Final = _reset_at(row.pop("reset_at"))
    assert row == expected, row
    info: Final = _budget_info(gateway, budget_id)
    assert {name: value for name, value in info.items() if name not in VOLATILE_INFO_FIELDS} == {
        **expected,
        **UNLOADED_RELATIONS,
    }, info
    assert VOLATILE_INFO_FIELDS <= info.keys(), info
    assert _reset_at(info["budget_reset_at"]) == reset_at, info
    if reset_after is None:
        assert reset_at is None, row
    else:
        assert reset_at is not None and reset_after < reset_at <= reset_after + reset_within, (reset_at, reset_after)


def _update(gateway: Gateway, body: dict[str, JsonValue]) -> None:
    response: Final = gateway.request("POST", "/budget/update", body)
    assert response.status_code == 200, response.text


def test_partial_update_keeps_sibling_budget_fields(gateway: Gateway) -> None:
    model_budget: Final[dict[str, JsonValue]] = {"openai/gpt-4o-mini": {"max_budget": 1, "budget_duration": "1d"}}
    with gateway.scenario() as scenario:
        created_at: Final = datetime.now(timezone.utc)
        budget_id: Final = scenario.budget(
            max_budget=10.0,
            soft_budget=5.0,
            tpm_limit=1000,
            rpm_limit=60,
            tpd_limit=100000,
            model_max_budget=model_budget,
            budget_duration="30d",
        )
        creator: Final = string_value(_budget_row(budget_id)["created_by"])
        expected: Final[dict[str, JsonValue]] = {
            "budget_id": budget_id,
            "max_budget": 10.0,
            "soft_budget": 5.0,
            "max_parallel_requests": None,
            "tpm_limit": 1000,
            "rpm_limit": 60,
            "tpd_limit": 100000,
            "model_max_budget": {"openai/gpt-4o-mini": {"max_budget": 1.0, "budget_duration": "1d"}},
            "budget_duration": "30d",
            "allowed_models": [],
            "temp_budget_increase": None,
            "temp_budget_expiry": None,
            "created_by": creator,
            "updated_by": creator,
        }
        _assert_budget(gateway, budget_id, expected, created_at, timedelta(days=30, minutes=5))

        before_partial: Final = datetime.now(timezone.utc)
        _update(gateway, {"budget_id": budget_id, "max_budget": 20.0})
        expected["max_budget"] = 20.0
        _assert_budget(gateway, budget_id, expected, created_at, timedelta(days=30, minutes=5))

        _update(
            gateway,
            {
                "budget_id": budget_id,
                "model_max_budget": {"openai/gpt-4o-mini": {"max_budget": 2, "budget_duration": "1d"}},
            },
        )
        expected["model_max_budget"] = {"openai/gpt-4o-mini": {"max_budget": 2.0, "budget_duration": "1d"}}
        _assert_budget(gateway, budget_id, expected, created_at, timedelta(days=30, minutes=5))

        _update(gateway, {"budget_id": budget_id, "soft_budget": None})
        expected["soft_budget"] = None
        _assert_budget(gateway, budget_id, expected, created_at, timedelta(days=30, minutes=5))

        before_hourly: Final = datetime.now(timezone.utc)
        _update(gateway, {"budget_id": budget_id, "budget_duration": "1h"})
        expected["budget_duration"] = "1h"
        _assert_budget(gateway, budget_id, expected, before_hourly, timedelta(hours=1, minutes=5))
        assert before_partial <= before_hourly

        _update(gateway, {"budget_id": budget_id, "budget_duration": None})
        expected["budget_duration"] = None
        _assert_budget(gateway, budget_id, expected, None, timedelta())

        pinned: Final = datetime(2030, 1, 1, tzinfo=timezone.utc)
        _update(gateway, {"budget_id": budget_id, "budget_duration": None, "budget_reset_at": pinned.isoformat()})
        _assert_budget(gateway, budget_id, expected, pinned - timedelta(microseconds=1), timedelta(microseconds=1))

        rejected: Final = gateway.request(
            "POST",
            "/budget/update",
            {"budget_id": budget_id, "model_max_budget": {"openai/gpt-4o-mini": {"max_budget": "lots"}}},
        )
        assert rejected.status_code == 422, rejected.text
        assert rejected.json() == {
            "detail": [
                {
                    "type": "float_parsing",
                    "loc": ["body", "model_max_budget", "openai/gpt-4o-mini", "max_budget"],
                    "msg": "Input should be a valid number, unable to parse string as a number",
                }
            ]
        }, rejected.text
        _assert_budget(gateway, budget_id, expected, pinned - timedelta(microseconds=1), timedelta(microseconds=1))


def test_model_max_budget_update_is_refused_by_the_handler_without_a_license(gateway: Gateway, tmp_path: Path) -> None:
    model_budget: Final[dict[str, JsonValue]] = {"openai/gpt-4o-mini": {"max_budget": 1, "budget_duration": "1d"}}
    with gateway.scenario() as scenario:
        budget_id: Final = scenario.budget(max_budget=10.0, model_max_budget=model_budget)
        before: Final = _budget_row(budget_id)
        with owned_proxy(gateway, tmp_path, {}, remove_environment=("LITELLM_LICENSE",)) as unlicensed:
            rejected: Final = unlicensed.request(
                "POST",
                "/budget/update",
                {"budget_id": budget_id, "model_max_budget": {"openai/gpt-4o-mini": {"max_budget": 2}}},
            )
            assert rejected.status_code == 400, rejected.text
            assert rejected.json() == {
                "detail": {
                    "error": "Invalid model_max_budget: You must have an enterprise license to set model_max_budget. "
                    "You must be a LiteLLM Enterprise user to use this feature. If you have a license please set "
                    "`LITELLM_LICENSE` in your env. Get a 7 day trial key here: https://www.litellm.ai/enterprise#trial. "
                    "\nPricing: https://www.litellm.ai/#pricing. Example of valid model_max_budget: "
                    "https://docs.litellm.ai/docs/proxy/users"
                }
            }, rejected.text
            assert _budget_row(budget_id) == before
            served: Final = unlicensed.request("POST", "/budget/update", {"budget_id": budget_id, "max_budget": 20.0})
            assert served.status_code == 200, served.text
            assert _budget_row(budget_id) == {**before, "max_budget": 20.0}
