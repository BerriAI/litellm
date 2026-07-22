"""Dispatching team metadata validator for the store_model_in_db e2e suite.

The suite runs one proxy with one config, so a single registered validator
dispatches to one of three independent implementations chosen per request via
the `_e2e_validator_impl` metadata key:

- `allowlist`: requires `cost_center` and checks it against a static set
- `http`: POSTs the metadata to the cost center service at
  `TEAM_METADATA_VALIDATION_SERVICE_URL`; transport errors raise (fail closed)
- `http_down`: like `http` but targets a closed port, proving the 503 path
- `immutable`: requires `cost_center` and forbids changing it once set

A request whose metadata carries no `_e2e_validator_impl` key is accepted
untouched, so the rest of the suite's team operations are unaffected. An
unknown impl value raises, which the proxy converts to the fail-closed 503.
"""

import os

import httpx

from litellm.proxy.management_helpers.team_metadata_validation import (
    TeamMetadataValidationPayload,
    TeamMetadataValidationResult,
)

ALLOWED_COST_CENTERS = frozenset({"CC-1001", "CC-1002"})
CLOSED_PORT_URL = "http://127.0.0.1:9/validate"


async def _validate_allowlist(payload: TeamMetadataValidationPayload) -> TeamMetadataValidationResult:
    cost_center = payload.metadata.get("cost_center")
    if cost_center is None:
        return TeamMetadataValidationResult(
            valid=False,
            error_message="cost_center is required in team metadata. Contact the FinOps team.",
        )
    if cost_center not in ALLOWED_COST_CENTERS:
        return TeamMetadataValidationResult(
            valid=False,
            error_message=f"Cost center {cost_center} is not recognized. Contact the FinOps team.",
        )
    return TeamMetadataValidationResult(valid=True)


async def _validate_via_http(payload: TeamMetadataValidationPayload, service_url: str) -> TeamMetadataValidationResult:
    async with httpx.AsyncClient(timeout=2.0) as client:
        response = await client.post(
            service_url,
            json={"operation": payload.operation, "metadata": payload.metadata},
        )
        response.raise_for_status()
        body = response.json()
    if body.get("ok") is True:
        return TeamMetadataValidationResult(valid=True)
    return TeamMetadataValidationResult(
        valid=False,
        error_message=body.get("reason", "Rejected by the cost center service."),
    )


class _ImmutableCostCenterValidator:
    def __init__(self, immutable_key: str = "cost_center") -> None:
        self.immutable_key = immutable_key

    async def __call__(self, payload: TeamMetadataValidationPayload) -> TeamMetadataValidationResult:
        current = payload.metadata.get(self.immutable_key)
        if current is None:
            return TeamMetadataValidationResult(
                valid=False,
                error_message=f"{self.immutable_key} is required in team metadata. Contact the FinOps team.",
            )
        if payload.operation == "update" and payload.existing_metadata is not None:
            prior = payload.existing_metadata.get(self.immutable_key)
            if prior is not None and prior != current:
                return TeamMetadataValidationResult(
                    valid=False,
                    error_message=(
                        f"{self.immutable_key} is immutable once set "
                        f"(stored: {prior}, requested: {current}). Contact the FinOps team."
                    ),
                )
        return TeamMetadataValidationResult(valid=True)


_IMMUTABLE_VALIDATOR = _ImmutableCostCenterValidator()


async def validate_team_metadata(
    payload: TeamMetadataValidationPayload,
) -> TeamMetadataValidationResult:
    impl = payload.metadata.get("_e2e_validator_impl")
    if impl is None:
        return TeamMetadataValidationResult(valid=True)
    if impl == "allowlist":
        return await _validate_allowlist(payload)
    if impl == "http":
        service_url = os.environ.get("TEAM_METADATA_VALIDATION_SERVICE_URL", "http://localhost:9414/validate")
        return await _validate_via_http(payload, service_url)
    if impl == "http_down":
        return await _validate_via_http(payload, CLOSED_PORT_URL)
    if impl == "immutable":
        return await _IMMUTABLE_VALIDATOR(payload)
    raise ValueError(f"unknown _e2e_validator_impl: {impl}")
