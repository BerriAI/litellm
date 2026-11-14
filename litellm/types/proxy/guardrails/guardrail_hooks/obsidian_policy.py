from pydantic import Field

from litellm.types.llms.base import LiteLLMBaseModel

from .base import GuardrailConfigModel


class ObsidianPolicyGuardrailConfigModel(GuardrailConfigModel[LiteLLMBaseModel]):
    api_key: str | None = Field(
        default=None,
        description="Obsidian gateway token. If not provided, the `OBSIDIAN_GATEWAY_TOKEN` environment variable is used.",
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "Obsidian Unified Evaluation endpoint. If not provided, the `OBSIDIAN_ENDPOINT` environment variable is "
            "used, then the Obsidian US evaluation host."
        ),
    )
    client_platform: str | None = Field(
        default=None,
        description=(
            "Name this gateway registers with Obsidian. If not provided, the `OBSIDIAN_CLIENT_PLATFORM` environment "
            "variable is used, then `litellm_gateway`. Names reserved by Obsidian's built-in integrations are rejected."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Obsidian Security Policy"
