from typing import Literal

from pydantic import Field

from .base import GuardrailConfigModel


class PromptSecurityGuardrailConfigModel(GuardrailConfigModel):
    api_key: str | None = Field(
        default=None,
        description="The API key for the Prompt Security guardrail. If not provided, the `PROMPT_SECURITY_API_KEY` environment variable is used.",
    )
    api_base: str | None = Field(
        default=None,
        description="The API base for the Prompt Security guardrail. If not provided, the `PROMPT_SECURITY_API_BASE` environment variable is used.",
    )
    file_sanitization_fail_open: bool = Field(
        default=True,
        description="Whether file sanitization timeouts allow the original file through instead of blocking the request.",
    )
    block_on_file_modify: bool = Field(
        default=True,
        description="Whether a file sanitization `modify` verdict blocks the request instead of replacing the file content.",
    )
    streaming_transform_mode: Literal["block_only", "incremental_diff"] | None = Field(
        default=None,
        description=(
            "How post_call `modify` verdicts reach a streaming client. `block_only` (default) streams the raw upstream "
            "chunks and only a `block` verdict ends the stream, so `modified_text` is dropped. `incremental_diff` "
            "buffers the whole response and sends the redacted text once the final verdict is in, so the first token "
            "arrives with the last, while a `block` verdict still ends the stream early. "
            "OpenAI chat completions streaming only."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Prompt Security"
