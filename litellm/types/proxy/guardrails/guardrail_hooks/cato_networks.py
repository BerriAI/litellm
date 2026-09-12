from pydantic import Field

from litellm.types.guardrails import GuardrailParamUITypes

from .base import GuardrailConfigModel


class CatoNetworksGuardrailConfigModel(GuardrailConfigModel):
    api_key: str | None = Field(
        default=None,
        description="The API key for the Cato Networks guardrail. If not provided, the `CATO_API_KEY` environment variable is checked.",
    )
    api_base: str | None = Field(
        default=None,
        description="The API base for the Cato Networks guardrail. Default is https://api.aisec.catonetworks.com. Also checks if the `CATO_API_BASE` environment variable is set.",
    )
    inspect_embeddings: bool | None = Field(
        default=False,
        description=(
            "Send /embeddings `input` to Cato Networks as user messages. Off by default because embedding "
            "input is documents being indexed, not a conversation."
        ),
        json_schema_extra={"ui_type": GuardrailParamUITypes.BOOL},  # mutable-ok: pydantic accepts only a dict here
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Cato Networks Guardrail"
