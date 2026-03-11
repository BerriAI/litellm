from typing import Optional

from pydantic import Field, model_validator

from litellm._logging import verbose_proxy_logger
from litellm.types.guardrails import GuardrailParamUITypes

from .base import GuardrailConfigModel


class ZscalerAIGuardConfigModel(GuardrailConfigModel):
    api_key: Optional[str] = Field(
        default=None,
        description=(
            "API key for Zscaler AI Guard authentication. "
            "If not provided, falls back to ZSCALER_AI_GUARD_API_KEY environment variable."
        ),
    )

    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Zscaler AI Guard API endpoint. Determines policy resolution behavior:\n"
            "• /execute-policy (default) - Requires explicit policy_id in configuration\n"
            "• /resolve-and-execute-policy - Infers policy from user-api-key-alias header\n"
            "Default: https://api.us1.zseclipse.net/v1/detection/execute-policy\n"
            "Falls back to ZSCALER_AI_GUARD_URL environment variable."
        ),
        json_schema_extra={
            "examples": [
                "https://api.us1.zseclipse.net/v1/detection/execute-policy",
                "https://api.us1.zseclipse.net/v1/detection/resolve-and-execute-policy",
            ]
        },
    )

    policy_id: Optional[int] = Field(
        default=None,
        description=(
            "Global policy ID for Zscaler AI Guard. Required when using /execute-policy endpoint.\n\n"
            "Set to 0 or leave empty when using /resolve-and-execute-policy with dynamic policy resolution.\n"
            "Falls back to ZSCALER_AI_GUARD_POLICY_ID environment variable."
        ),
        json_schema_extra={
            "ui_hint": "conditional_required",
            "condition": "Required when api_base ends with /execute-policy",
        },
    )

    send_user_api_key_alias: Optional[bool] = Field(
        default=False,
        description=(
            "Send user API key alias in request headers as 'user-api-key-alias'. "
            "CRITICAL when using /resolve-and-execute-policy endpoint - the policy is inferred from this value. "
            "Also useful for tracking/auditing with /execute-policy endpoint."
        ),
        json_schema_extra={
            "ui_type": GuardrailParamUITypes.BOOL,
            "ui_hint": "recommended_when",
            "condition": "Recommended when api_base ends with /resolve-and-execute-policy",
        },
    )

    send_user_api_key_user_id: Optional[bool] = Field(
        default=False,
        description=(
            "Send user API key user_id in request headers as 'user-api-key-user-id'. "
            "Enables user-level tracking and analytics in Zscaler AI Guard."
        ),
        json_schema_extra={"ui_type": GuardrailParamUITypes.BOOL},
    )

    send_user_api_key_team_id: Optional[bool] = Field(
        default=False,
        description=(
            "Send user API key team_id in request headers as 'user-api-key-team-id'. "
            "Enables team-level tracking and analytics in Zscaler AI Guard."
        ),
        json_schema_extra={"ui_type": GuardrailParamUITypes.BOOL},
    )

    @model_validator(mode="after")
    def validate_endpoint_configuration(self) -> "ZscalerAIGuardConfigModel":
        """
        Validate configuration consistency between api_base and other fields.
        Provides warnings but doesn't block (since env vars might provide values).
        """
        import os

        # Resolve actual api_base value (including env fallback)
        api_base = self.api_base or os.getenv(
            "ZSCALER_AI_GUARD_URL",
            "https://api.us1.zseclipse.net/v1/detection/execute-policy",
        )

        # Resolve actual policy_id value
        policy_id = self.policy_id
        if policy_id is None:
            env_policy = os.getenv("ZSCALER_AI_GUARD_POLICY_ID")
            if env_policy:
                try:
                    policy_id = int(env_policy)
                except ValueError:
                    verbose_proxy_logger.warning(
                        f"ZSCALER_AI_GUARD_POLICY_ID env var is not a valid integer: {env_policy}"
                    )

        # Check for configuration issues
        assert api_base is not None  # always set via env default above
        is_resolve_policy = api_base.endswith("/resolve-and-execute-policy")
        is_execute_policy = api_base.endswith("/execute-policy") and not is_resolve_policy

        # Scenario A: execute-policy without policy_id
        if is_execute_policy and (policy_id is None or policy_id < 1):
            verbose_proxy_logger.warning(
                "Using /execute-policy endpoint without a valid policy_id. "
                "Ensure ZSCALER_AI_GUARD_POLICY_ID environment variable is set, "
                "or provide policy_id via request/key/team metadata."
            )

        # Scenario B: resolve-and-execute-policy without user_api_key_alias
        if is_resolve_policy and not self.send_user_api_key_alias:
            verbose_proxy_logger.warning(
                "Using /resolve-and-execute-policy endpoint without send_user_api_key_alias=true. "
                "The endpoint requires user-api-key-alias header to resolve the policy. "
                "Set send_user_api_key_alias to true or ensure the header is sent via other means."
            )

        return self

    @staticmethod
    def ui_friendly_name() -> str:
        return "Zscaler AI Guard"
