from __future__ import annotations

import functools
import json
import os
from collections.abc import Mapping
from hashlib import sha256
from typing import Final

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import BaseModel, ConfigDict

from integration._support.client import JSON_OBJECT, Scenario, eventually, object_value, string_value
from integration._support.database import read_rows
from integration._support.upstream import delete_scenario, register_scenario
from integration.cost_calculation.cost_matrix import Case, FrontierModel


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
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    metadata: CostMetadata | None = None

    @property
    def breakdown(self) -> CostBreakdown:
        assert self.metadata is not None and self.metadata.cost_breakdown is not None
        return self.metadata.cost_breakdown


def approx_equal(actual: float, expected: float) -> bool:
    return abs(actual - expected) <= max(1e-9, abs(expected) * 1e-2)


def assert_total_is_sum_of_components(row: CostRow, context: str) -> None:
    breakdown: Final = row.breakdown
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
    return parsed if parsed.metadata and parsed.metadata.cost_breakdown else None


def poll_cost_row(key: str) -> CostRow:
    digest: Final = sha256(key.encode()).hexdigest()

    def read() -> CostRow | None:
        rows: Final = read_rows(
            'SELECT spend, metadata, prompt_tokens, completion_tokens FROM "LiteLLM_SpendLogs" WHERE api_key=%s',
            (digest,),
        )
        return next((parsed for row in rows if (parsed := _row(row)) is not None), None)

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


def register_scenario_deployment(
    scenario: Scenario,
    model: FrontierModel,
    case: Case,
    marker: str,
) -> str:
    control_url: Final = os.environ["INTEGRATION_UPSTREAM_URL"].rstrip("/")
    sidecar_scenario: Final = case.scenario(
        scenario_id=f"sc-{marker}", model=model, text=f"scripted answer {marker}"
    )
    handle: Final = register_scenario(sidecar_scenario)
    scenario.cleanups.callback(delete_scenario, handle)
    model_name: Final = f"{model.model_name}-{marker}"
    parameters: Final = {
        "model": model.litellm_model,
        "api_key": model.api_key,
        "api_base": handle.api_base(),
        **model.litellm_params,
        **(
            {"vertex_credentials": _vertex_service_account_json(control_url)}
            if model.wire == "vertex_generate"
            else {}
        ),
    }
    created: Final = scenario.gateway.post(
        "/model/new",
        JSON_OBJECT.validate_python({
            "model_name": model_name,
            "litellm_params": parameters,
            "model_info": {"base_model": model.base_model},
        }),
    )
    identity: Final = string_value(object_value(created["model_info"])["id"])
    scenario.cleanups.callback(scenario.delete_model, identity)
    return model_name
