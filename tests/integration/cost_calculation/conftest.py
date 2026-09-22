from __future__ import annotations

import functools
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Final

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from integration._support.client import JSON_OBJECT, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.upstream import ScenarioHandle, delete_scenario, register_scenario
from integration.cost_calculation.cost_tracking_case import CostTrackingTestCase, StoredResponse
from pydantic import BaseModel, ConfigDict


class CostBreakdown(BaseModel):
    model_config = ConfigDict(extra="ignore")

    input_cost: float | None = None
    output_cost: float | None = None
    cache_read_cost: float | None = None
    cache_creation_cost: float | None = None
    reasoning_cost: float | None = None
    tool_usage_cost: float | None = None
    total_cost: float | None = None
    service_tier: str | None = None


class CostMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cost_breakdown: CostBreakdown | None = None


class CostRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    spend: float | None = None
    status: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model_id: str | None = None
    call_type: str | None = None
    metadata: CostMetadata | None = None

    @property
    def breakdown(self) -> CostBreakdown | None:
        return self.metadata.cost_breakdown if self.metadata is not None else None


class FailureRow(BaseModel):
    model_config = ConfigDict(extra="ignore")

    spend: float
    status: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class DailySpend(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    spend: float
    prompt_tokens: int
    completion_tokens: int
    api_requests: int


class Rollups(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key_spend: float
    team_spend: float
    user_spend: float
    end_user_spend: float
    daily_user: DailySpend
    daily_team: DailySpend


def approx_equal(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)


def assert_total_is_sum_of_components(row: CostRow, breakdown: CostBreakdown, context: str) -> None:
    total: Final = sum(
        cost or 0.0
        for cost in (breakdown.input_cost, breakdown.output_cost, breakdown.tool_usage_cost)
    )
    assert breakdown.total_cost is not None and approx_equal(breakdown.total_cost, total), (
        f"{context}: total_cost {breakdown.total_cost} != input_cost {breakdown.input_cost} "
        f"+ output_cost {breakdown.output_cost} + tool_usage_cost {breakdown.tool_usage_cost} "
        f"(sum {total})"
    )
    assert row.spend is not None and approx_equal(row.spend, breakdown.total_cost), (
        f"{context}: row spend {row.spend} != breakdown total_cost {breakdown.total_cost}"
    )


def _row(value: Mapping[str, object]) -> CostRow | None:
    metadata_value: Final = value.get("metadata")
    metadata: Final = json.loads(metadata_value) if isinstance(metadata_value, str) else metadata_value
    parsed: Final = CostRow.model_validate({**value, "metadata": metadata})
    return parsed if parsed.metadata is not None or (parsed.spend is not None and parsed.status is not None) else None


def poll_cost_row(key: str) -> CostRow:
    digest: Final = sha256(key.encode()).hexdigest()

    def read() -> CostRow | None:
        rows: Final = read_rows(
            'SELECT spend, status, metadata, prompt_tokens, completion_tokens, model_id, call_type '
            'FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (digest,),
        )
        return next((parsed for row in rows if (parsed := _row(row)) is not None), None)

    result: Final = eventually(read, lambda row: row is not None, seconds=60)
    assert result is not None
    return result


def read_rows_now(key: str) -> tuple[CostRow, ...]:
    digest: Final = sha256(key.encode()).hexdigest()
    rows: Final = read_rows(
        'SELECT spend, status, metadata, prompt_tokens, completion_tokens, model_id, call_type '
        'FROM "LiteLLM_SpendLogs" WHERE api_key=%s ORDER BY "startTime"',
        (digest,),
    )
    return tuple(parsed for row in rows if (parsed := _row(row)) is not None)


def poll_rows(key: str, count: int) -> tuple[CostRow, ...]:
    return poll_rows_where(key, count, lambda _row: True)


def poll_rows_where(
    key: str,
    count: int,
    predicate: Callable[[CostRow], bool],
) -> tuple[CostRow, ...]:
    result: Final = eventually(
        lambda: tuple(row for row in read_rows_now(key) if predicate(row)),
        lambda rows: len(rows) >= count,
        seconds=60,
    )
    return result


def poll_rollups(
    key: str,
    team_id: str,
    user_id: str,
    end_user_id: str,
    target_spend: float,
    target_requests: int,
) -> Rollups:
    digest: Final = sha256(key.encode()).hexdigest()

    def read() -> Rollups | None:
        key_rows: Final = read_rows(
            'SELECT spend FROM "LiteLLM_VerificationToken" WHERE token=%s',
            (digest,),
        )
        team_rows: Final = read_rows(
            'SELECT spend FROM "LiteLLM_TeamTable" WHERE team_id=%s',
            (team_id,),
        )
        user_rows: Final = read_rows(
            'SELECT spend FROM "LiteLLM_UserTable" WHERE user_id=%s',
            (user_id,),
        )
        end_user_rows: Final = read_rows(
            'SELECT spend FROM "LiteLLM_EndUserTable" WHERE user_id=%s',
            (end_user_id,),
        )
        daily_user_rows: Final = read_rows(
            'SELECT spend, prompt_tokens, completion_tokens, api_requests '
            'FROM "LiteLLM_DailyUserSpend" WHERE user_id=%s AND api_key=%s AND date=CURRENT_DATE::text',
            (user_id, digest),
        )
        daily_team_rows: Final = read_rows(
            'SELECT spend, prompt_tokens, completion_tokens, api_requests '
            'FROM "LiteLLM_DailyTeamSpend" WHERE team_id=%s AND api_key=%s AND date=CURRENT_DATE::text',
            (team_id, digest),
        )
        if not all((key_rows, team_rows, user_rows, end_user_rows, daily_user_rows, daily_team_rows)):
            return None
        rollups: Final = Rollups(
            key_spend=float(key_rows[0]["spend"]),
            team_spend=float(team_rows[0]["spend"]),
            user_spend=float(user_rows[0]["spend"]),
            end_user_spend=float(end_user_rows[0]["spend"]),
            daily_user=DailySpend.model_validate(daily_user_rows[0]),
            daily_team=DailySpend.model_validate(daily_team_rows[0]),
        )
        return rollups

    def settled(value: Rollups | None) -> bool:
        return value is not None and all(
            (
                approx_equal(value.key_spend, target_spend),
                approx_equal(value.team_spend, target_spend),
                approx_equal(value.user_spend, target_spend),
                approx_equal(value.end_user_spend, target_spend),
                approx_equal(value.daily_user.spend, target_spend),
                approx_equal(value.daily_team.spend, target_spend),
                value.daily_user.api_requests == target_requests,
                value.daily_team.api_requests == target_requests,
            )
        )

    result: Final = eventually(
        read,
        settled,
        seconds=20,
        return_last_on_timeout=True,
    )
    assert result is not None
    return result


def poll_failure_row(key: str) -> FailureRow:
    digest: Final = sha256(key.encode()).hexdigest()

    def read() -> FailureRow | None:
        rows: Final = read_rows(
            'SELECT spend, status, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (digest,),
        )
        return next(
            (
                parsed
                for row in rows
                if (parsed := FailureRow.model_validate(row)).status == "failure"
            ),
            None,
        )

    result: Final = eventually(read, lambda row: row is not None, seconds=60)
    assert result is not None
    return result


@functools.cache
def _vertex_private_key_pem() -> str:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048).private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ).decode()


def _vertex_service_account_json(url: str) -> str:
    return json.dumps(
        {
            "type": "service_account",
            "project_id": "cc-scripted-project",
            "private_key_id": "scripted",
            "private_key": _vertex_private_key_pem(),
            "client_email": "scripted@cc-scripted-project.iam.gserviceaccount.com",
            "client_id": "0",
            "auth_uri": f"{url}/_oauth/authorize",
            "token_uri": f"{url}/_oauth/token",
        }
    )


@dataclass(frozen=True, slots=True)
class RegisteredDeployment:
    model_name: str
    identity: str
    handle: ScenarioHandle


def register_scenario_deployment(
    scenario: Scenario,
    case: CostTrackingTestCase,
    marker: str,
    key: str,
    *,
    response: StoredResponse | None = None,
    marker_suffix: str = "",
) -> RegisteredDeployment:
    control_url: Final = os.environ["INTEGRATION_UPSTREAM_URL"].rstrip("/")
    run_marker: Final = sha256(key.encode()).hexdigest()[:12]
    handle: Final = register_scenario(
        f"sc-{marker}{marker_suffix}-{run_marker}",
        case.response if response is None else response,
    )
    scenario.cleanups.callback(delete_scenario, handle)
    registered_model_name: Final = f"cost-{marker}{marker_suffix}-{run_marker}"
    parameters: Final = {
        "model": case.litellm_model,
        "api_key": case.api_key,
        "api_base": handle.api_base(),
        **case.litellm_params,
        **(
            {
                key: value
                for key, value in (
                    ("input_cost_per_token", case.deployment.input_cost_per_token),
                    ("output_cost_per_token", case.deployment.output_cost_per_token),
                )
                if value is not None
            }
            if case.deployment is not None
            else {}
        ),
        **(
            {"vertex_credentials": _vertex_service_account_json(control_url)}
            if case.rates.litellm_provider.startswith("vertex_ai")
            else {}
        ),
    }
    created: Final = scenario.gateway.post(
        "/model/new",
        JSON_OBJECT.validate_python({
            "model_name": registered_model_name,
            "litellm_params": parameters,
            "model_info": (
                {"base_model": case.base_model}
                if case.base_model is not None
                else {}
            ),
        }),
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return RegisteredDeployment(model_name=registered_model_name, identity=identity, handle=handle)
