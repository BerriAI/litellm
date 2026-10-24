from collections.abc import Mapping, Sequence
from typing import Annotated

from pydantic import BaseModel, Field
from typing_extensions import ReadOnly, Required, TypedDict

from .base import GuardrailConfigModel


class RealmLabsChatMessage(TypedDict):
    """A plain-text chat turn sent to MLS."""

    role: ReadOnly[str]
    content: ReadOnly[str]


class RealmLabsGuardrailRequest(TypedDict):
    messages: ReadOnly[Sequence[Mapping[str, object]]]
    probes: ReadOnly[Sequence[str] | str]
    pii: ReadOnly[bool]
    enable_thinking: ReadOnly[bool]


class RealmLabsProbeResult(TypedDict, total=False):
    """One classifier probe verdict.

    ``prob`` is compared against the guardrail's own ``hazard_threshold``, not the ``threshold`` MLS reports.
    """

    probe: ReadOnly[Required[Annotated[str, Field(strict=True, min_length=1)]]]
    prob: ReadOnly[Required[Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]]]
    threshold: ReadOnly[float | None]
    decision: ReadOnly[bool | None]
    role_mismatch: ReadOnly[Annotated[bool, Field(strict=True)] | None]


class RealmLabsPIISpan(TypedDict, total=False):
    """One detected PII span.

    ``start``/``end`` index MLS's rendering of the whole conversation, not a single message, so they are not
    used for masking; ``text`` is matched within each message instead.
    """

    type: ReadOnly[Annotated[str, Field(strict=True, min_length=1)]]
    text: ReadOnly[str | None]
    start: ReadOnly[int | None]
    end: ReadOnly[int | None]


class RealmLabsGuardrailResponse(TypedDict, total=False):
    """Response body of ``POST {api_base}/guardrail``. MLS is stateless, so it carries no turn id."""

    results: ReadOnly[Required[Sequence[RealmLabsProbeResult]]]
    focal_role: ReadOnly[str | None]
    pii_spans: ReadOnly[Required[Sequence[RealmLabsPIISpan]]]


class RealmLabsGuardrailOptionalParams(BaseModel):
    """Nested tuning settings; explicitly supplied non-null values override the top-level settings."""

    probes: Sequence[str] | str | None = Field(
        default=None,
        description="Classifier probes to run: a list of names or 'all'. Overrides top-level probes when supplied.",
    )
    hazard_threshold: float | None = Field(
        default=None,
        description="Block hazard scores strictly above this value. Overrides top-level hazard_threshold when supplied.",
    )
    pii: bool | None = Field(
        default=None,
        description="Whether to run PII detection. Overrides top-level pii when supplied.",
    )
    pii_mask: bool | None = Field(
        default=None,
        description="Mask detected PII when true, otherwise block. Overrides top-level pii_mask when supplied.",
    )
    block_on_error: bool | None = Field(
        default=None,
        description="Whether to block when MLS fails. Overrides top-level block_on_error when supplied.",
    )

    enable_thinking: bool | None = Field(
        default=False,
        description="Whether MLS should render the chat template in thinking mode. Defaults to False.",
    )

    timeout: float | None = Field(
        default=15.0,
        description="Timeout in seconds for the MLS request. Defaults to 15.",
    )


class RealmLabsGuardrailConfigModel(GuardrailConfigModel[RealmLabsGuardrailOptionalParams]):
    """Settings accepted under ``litellm_params`` for ``guardrail: realmlabs``."""

    api_key: str | None = Field(
        default=None,
        description=(
            "API key for the RealmLabs MLS guardrail endpoint, sent as a bearer token. "
            "If not provided, the REALMLABS_API_KEY environment variable is used."
        ),
    )
    api_base: str | None = Field(
        default=None,
        description=(
            "Base URL of the RealmLabs MLS deployment. The /guardrail path is "
            "appended automatically. Defaults to https://mls.realmlabs.ai, and falls "
            "back to the REALMLABS_API_BASE environment variable."
        ),
    )
    probes: list[str] | str | None = Field(  # mutable-ok: the admin UI renders only list[...] fields as array inputs
        default=None,
        description=(
            'Which classifier probes to run: a list of probe names, or "all". '
            'Defaults to ["hazard_prompt"] - the only probe whose score this '
            "guardrail enforces. An unknown probe name makes MLS return 404."
        ),
    )
    hazard_threshold: float | None = Field(
        default=None,
        description=(
            "Block the request when the hazard_prompt probe scores strictly above this "
            "value. Defaults to 0.703, the threshold MLS reports for that probe. Note "
            'the probe also responds to instruction-style phrasing such as "repeat this '
            'back verbatim", so raise this if benign traffic is being blocked.'
        ),
    )
    pii: bool | None = Field(
        default=None,
        description=("Whether to run MLS's PII detection head. Defaults to True."),
    )
    pii_mask: bool | None = Field(
        default=None,
        description=(
            "What to do with detected PII. True (default) rewrites each span as its "
            'type in brackets, e.g. "My name is Alex" -> "My name is [name]", and '
            "lets the request through. False blocks the request instead."
        ),
    )
    block_on_error: bool | None = Field(
        default=None,
        description=(
            "Whether to block the request when MLS is unreachable or returns an "
            "unreadable response. Defaults to False (fail open), so an MLS outage does "
            "not take the gateway down with it. Set to True to fail closed."
        ),
    )
    enable_thinking: bool | None = Field(
        default=None,
        description="Whether MLS should render the chat template in thinking mode. Defaults to False.",
    )

    @staticmethod
    def ui_friendly_name() -> str:
        """Name the admin UI shows for this guardrail."""
        return "RealmLabs MLS"
