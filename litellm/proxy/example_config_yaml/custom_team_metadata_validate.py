"""Example validator for `general_settings.custom_team_metadata_validate`.

Wire it up in the proxy config:

```yaml
general_settings:
  custom_team_metadata_validate: custom_team_metadata_validate.validate_team_metadata
  team_metadata_validation_timeout: 5
  team_metadata_validation_error_message: "Validation service unavailable, contact your admin."
```

Return `valid=False` with an `error_message` to reject the write with that
message (HTTP 400). Raise any exception (for example, when the upstream
validation service is unreachable) to fail closed with the generic
`team_metadata_validation_error_message` (HTTP 503).
"""

from litellm.proxy.management_helpers.team_metadata_validation import (
    TeamMetadataValidationPayload,
    TeamMetadataValidationResult,
)

VALID_COST_CENTERS = {"CC-1001", "CC-1002", "CC-2001"}


async def validate_team_metadata(
    payload: TeamMetadataValidationPayload,
) -> TeamMetadataValidationResult:
    cost_center = payload.metadata.get("cost_center")
    if cost_center is None:
        return TeamMetadataValidationResult(
            valid=False,
            error_message="Team metadata must include a cost_center. Contact the FinOps team.",
        )
    if cost_center not in VALID_COST_CENTERS:
        return TeamMetadataValidationResult(
            valid=False,
            error_message=f"Cost center {cost_center} is not recognized. Contact the FinOps team.",
        )
    return TeamMetadataValidationResult(valid=True)
