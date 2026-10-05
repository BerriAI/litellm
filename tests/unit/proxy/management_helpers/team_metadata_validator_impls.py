"""Three independent `custom_team_metadata_validate` implementations.

Used by the matrix tests in `test_team_metadata_validation.py` and loadable
directly from a proxy config via `get_instance_fn` for live verification:

- `validate_allowlist`: plain async function; requires `cost_center` and
  checks it against a static allowlist.
- `validate_via_http`: async function that POSTs the metadata to an external
  validation service (`TEAM_METADATA_VALIDATION_SERVICE_URL`); any transport
  error or non-2xx response raises, exercising the fail-closed path.
- `IMMUTABLE_COST_CENTER_VALIDATOR`: class instance with an async
  `__call__`; requires `cost_center` and forbids changing it once set, using
  `existing_metadata` and `operation`.
"""

import os

import httpx

from litellm.proxy.management_helpers.team_metadata_validation import (
    TeamMetadataValidationPayload,
    TeamMetadataValidationResult,
)

ALLOWED_COST_CENTERS = frozenset({"CC-1001", "CC-1002"})
DEFAULT_SERVICE_URL = "http://localhost:9414/validate"


async def validate_allowlist(
    payload: TeamMetadataValidationPayload,
) -> TeamMetadataValidationResult:
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


async def validate_via_http(
    payload: TeamMetadataValidationPayload,
) -> TeamMetadataValidationResult:
    service_url = os.environ.get("TEAM_METADATA_VALIDATION_SERVICE_URL", DEFAULT_SERVICE_URL)
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


class ImmutableCostCenterValidator:
    def __init__(self, immutable_key: str = "cost_center") -> None:
        self.immutable_key = immutable_key

    async def __call__(
        self,
        payload: TeamMetadataValidationPayload,
    ) -> TeamMetadataValidationResult:
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


IMMUTABLE_COST_CENTER_VALIDATOR = ImmutableCostCenterValidator()
