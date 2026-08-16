from typing import Any, Dict, Literal

from pydantic import BaseModel, Field

from .base import GuardrailConfigModel


class CompresrGuardrailOptionalParams(BaseModel):
    """Optional tuning knobs for the Compresr guardrail."""

    target_compression_ratio: float | None = Field(
        default=None,
        description=(
            "Compression strength. 0-1 is the fraction of tokens to remove "
            "(0.5 = remove ~50%, the default); a value >1 is an Nx reduction "
            "factor (e.g. 4 = ~4x smaller)."
        ),
    )
    coarse: bool | None = Field(
        default=None,
        description=("Paragraph-level compression (default, faster) instead of token-level (finer-grained)."),
    )
    min_chars_to_compress: int | None = Field(
        default=None,
        description=("Skip messages whose text is shorter than this many characters. Defaults to 500."),
    )
    compress_tool_outputs: bool | None = Field(
        default=None,
        description=("Compress tool/function result messages (search hits, RAG chunks, API dumps). Defaults to True."),
    )
    compress_system: bool | None = Field(
        default=None,
        description="Also compress system messages. Defaults to False.",
    )
    compress_history: bool | None = Field(
        default=None,
        description="Also compress prior (non-last) user messages. Defaults to False.",
    )
    compress_last_user: bool | None = Field(
        default=None,
        description=(
            "Also compress the last user message. The query sent to Compresr "
            "is always the original verbatim text. Defaults to False."
        ),
    )
    enable_retrieval: bool | None = Field(
        default=None,
        description=(
            "Make compression recoverable: inject a `compresr_retrieve` tool "
            "so the model can fetch the original content behind a compression "
            "marker via the agentic loop. Defaults to True. Set to False (or "
            "run the proxy with --workers 1) for multi-worker deployments: "
            "the recovery store is per-process, so pre-call and retrieval hooks "
            "on different workers cannot see each other's originals."
        ),
    )
    max_bytes_per_call: int | None = Field(
        default=None,
        description=(
            "Cap on aggregate bytes of stored originals per litellm_call_id. "
            "When a call exceeds this, oldest entries are evicted so the "
            "in-process store cannot grow without bound. Defaults to 10 MiB."
        ),
    )
    allow_bypass_header: bool | None = Field(
        default=None,
        description=(
            "Honor the `x-compresr-bypass: true` request header to skip "
            "compression for a single call. Off by default because the "
            "header is caller-settable; enable only on trusted deployments."
        ),
    )
    dynamic: bool | None = Field(
        default=None,
        description=(
            "latte_v2 only. Let the server choose the compression amount per input "
            "(Kneedle elbow) instead of using target_compression_ratio. Defaults to True."
        ),
    )
    dynamic_min_ratio: float | None = Field(
        default=None,
        description=(
            "latte_v2 only. Floor on the adaptive ratio when `dynamic` is on. "
            "Unset lets the server default apply (~1.5)."
        ),
    )
    dynamic_max_ratio: float | None = Field(
        default=None,
        description=(
            "latte_v2 only. Ceiling on the adaptive ratio when `dynamic` is on. "
            "Unset lets the server default apply (~10.0)."
        ),
    )
    compression_params: Dict[str, Any] | None = Field(
        default=None,
        description=(
            "Passthrough of extra parameters forwarded verbatim in the Compresr "
            "compress payload (e.g. `heuristic_chunking`, or any newer knob), so "
            "a new Compresr feature works without a guardrail update. The named "
            "fields above take precedence on collision."
        ),
    )


class CompresrGuardrailConfigModel(GuardrailConfigModel[CompresrGuardrailOptionalParams]):
    api_key: str | None = Field(
        default=None,
        description=("Compresr API key. Falls back to the COMPRESR_API_KEY env var."),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "Base URL of the Compresr API. Falls back to the COMPRESR_API_BASE "
            "env var, then https://api.compresr.ai. Point at your internal "
            "service URL for on-prem deployments."
        ),
    )
    model: str | None = Field(
        default=None,
        description=(
            "Compresr compression model (not the LLM). Defaults to 'latte_v2', the query-aware compression model."
        ),
    )
    unreachable_fallback: Literal["fail_closed", "fail_open"] = Field(
        default="fail_closed",
        description=(
            "Behavior when the Compresr compression service is unreachable or errors. "
            "'fail_closed' raises an error (default). 'fail_open' logs a critical error and "
            "forwards the request uncompressed instead of blocking it."
        ),
    )

    @staticmethod
    def ui_friendly_name() -> str:
        return "Compresr (context compression)"
