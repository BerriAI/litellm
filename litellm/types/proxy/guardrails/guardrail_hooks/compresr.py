from typing import Literal, Optional

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class CompresrGuardrailOptionalParams(BaseModel):
    """Optional tuning knobs for the Compresr guardrail."""

    target_compression_ratio: Optional[float] = Field(
        default=None,
        description=(
            "Compression strength. 0-1 is the fraction of tokens to remove "
            "(0.5 = remove ~50%, the default); a value >1 is an Nx reduction "
            "factor (e.g. 4 = ~4x smaller)."
        ),
    )
    coarse: Optional[bool] = Field(
        default=None,
        description=("Paragraph-level compression (default, faster) instead of token-level (finer-grained)."),
    )
    min_chars_to_compress: Optional[int] = Field(
        default=None,
        description=("Skip messages whose text is shorter than this many characters. Defaults to 500."),
    )
    compress_tool_outputs: Optional[bool] = Field(
        default=None,
        description=("Compress tool/function result messages (search hits, RAG chunks, API dumps). Defaults to True."),
    )
    compress_system: Optional[bool] = Field(
        default=None,
        description="Also compress system messages. Defaults to False.",
    )
    compress_history: Optional[bool] = Field(
        default=None,
        description="Also compress prior (non-last) user messages. Defaults to False.",
    )
    compress_last_user: Optional[bool] = Field(
        default=None,
        description=(
            "Also compress the last user message. The query sent to Compresr "
            "is always the original verbatim text. Defaults to False."
        ),
    )
    enable_retrieval: Optional[bool] = Field(
        default=None,
        description=(
            "Make compression recoverable: inject a `compresr_retrieve` tool "
            "so the model can fetch the original content behind a compression "
            "marker via the agentic loop. Defaults to True."
        ),
    )
    max_bytes_per_call: Optional[int] = Field(
        default=None,
        description=(
            "Cap on aggregate bytes of stored originals per litellm_call_id. "
            "When a call exceeds this, oldest entries are evicted so the "
            "in-process store cannot grow without bound. Defaults to 10 MiB."
        ),
    )
    allow_bypass_header: Optional[bool] = Field(
        default=None,
        description=(
            "Honor the `x-compresr-bypass: true` request header to skip "
            "compression for a single call. Off by default because the "
            "header is caller-settable; enable only on trusted deployments."
        ),
    )


class CompresrGuardrailConfigModel(GuardrailConfigModel[CompresrGuardrailOptionalParams]):
    api_key: Optional[str] = Field(
        default=None,
        description=("Compresr API key. Falls back to the COMPRESR_API_KEY env var."),
    )
    api_base: Optional[str] = Field(
        default=None,
        description=(
            "Base URL of the Compresr API. Falls back to the COMPRESR_API_BASE "
            "env var, then https://api.compresr.ai. Point at your internal "
            "service URL for on-prem deployments."
        ),
    )
    model: Optional[str] = Field(
        default=None,
        description=(
            "Compresr compression model (not the LLM). Defaults to 'latte_v2', the query-aware compression model."
        ),
    )
    unreachable_fallback: Optional[Literal["fail_open", "fail_closed"]] = Field(
        default=None,
        description=(
            "What to do when the Compresr service is unreachable. "
            "'fail_closed' (default) returns HTTP 502 to the caller. "
            "'fail_open' logs a critical warning and forwards the request uncompressed."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Compresr (context compression)"
