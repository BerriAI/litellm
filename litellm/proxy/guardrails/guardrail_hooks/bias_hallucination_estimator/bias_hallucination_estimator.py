from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclasses_field
from datetime import datetime, timezone
from typing import (
    TYPE_CHECKING,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    Union,
    cast,
)

from pydantic import BaseModel

from litellm.exceptions import GuardrailRaisedException
from litellm.integrations.custom_guardrail import (
    CustomGuardrail,
    log_guardrail_information,
)
from litellm.types.guardrails import GuardrailEventHooks, Mode
from litellm.types.proxy.guardrails.guardrail_hooks.base import GuardrailConfigModel
from litellm.types.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator import (
    BiasAnalysis,
    HallucinationAnalysis,
    RiskScore,
)
from litellm.types.utils import (
    GenericGuardrailAPIInputs,
    GuardrailStatus,
    GuardrailTracingDetail,
)

from .estimator_core import BiasDetector, HallucinationDetector
from .risk_scorer import RiskScorer, RiskThresholds, RiskWeights

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

GUARDRAIL_PROVIDER = "litellm_native"
_LOG_EXCLUDED_FIELDS: frozenset[str] = frozenset(
    {"examples", "unsourced_claims", "missing_citations", "fabricated_specificity"}
)
GuardrailEventHookInput = Union[
    str,
    Sequence[str],
    Mode,
]
NormalizedGuardrailEventHook = Union[
    GuardrailEventHooks, list[GuardrailEventHooks], Mode
]


class FunctionLike(Protocol):
    name: str | None
    arguments: str


class ToolCallLike(Protocol):
    function: FunctionLike


@dataclass(frozen=True, slots=True)
class TextRiskAnalysis:
    text: str
    bias: BiasAnalysis
    hallucination: HallucinationAnalysis
    risk: RiskScore


@dataclass
class GuardrailBehaviorConfig:
    """Controls what the guardrail checks and how it reacts to violations."""

    block_on_high_risk: bool = True
    log_only: bool = False
    check_request: bool = False
    check_response: bool = True
    violation_message: str | None = None


@dataclass
class GuardrailSessionConfig:
    """Session-level routing and messaging settings."""

    mask_request_content: bool = False
    mask_response_content: bool = False
    violation_message_template: str | None = None
    end_session_after_n_fails: int | None = None
    on_violation: str | None = None
    realtime_violation_message: str | None = None
    on_sensitive_data: str | None = None
    sensitive_data_route_to_model: str | None = None
    sticky_session_routing: bool = True


@dataclass
class GuardrailConfig:
    """Top-level configuration bundle for the bias/hallucination guardrail."""

    thresholds: RiskThresholds = dataclasses_field(default_factory=RiskThresholds)
    weights: RiskWeights = dataclasses_field(default_factory=RiskWeights)
    behavior: GuardrailBehaviorConfig = dataclasses_field(
        default_factory=GuardrailBehaviorConfig
    )
    session: GuardrailSessionConfig = dataclasses_field(
        default_factory=GuardrailSessionConfig
    )


class BiasHallucinationEstimatorGuardrail(CustomGuardrail):
    def __init__(
        self,
        *,
        guardrail_name: str | None = None,
        guardrail_id: str | None = None,
        event_hook: GuardrailEventHookInput | None = None,
        config: GuardrailConfig | None = None,
    ) -> None:
        _config = config or GuardrailConfig()
        _thresholds = _config.thresholds
        _weights = _config.weights
        _behavior = _config.behavior
        _session = _config.session
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            guardrail_name=guardrail_name,
            supported_event_hooks=[
                GuardrailEventHooks.pre_call,
                GuardrailEventHooks.post_call,
            ],
            event_hook=self._normalize_event_hook(event_hook)
            or GuardrailEventHooks.post_call,
            mask_request_content=_session.mask_request_content,
            mask_response_content=_session.mask_response_content,
            violation_message_template=_session.violation_message_template,
            end_session_after_n_fails=_session.end_session_after_n_fails,
            on_violation=_session.on_violation,
            realtime_violation_message=_session.realtime_violation_message,
            on_sensitive_data=_session.on_sensitive_data,
            sensitive_data_route_to_model=_session.sensitive_data_route_to_model,
            sticky_session_routing=_session.sticky_session_routing,
        )
        self.guardrail_provider = GUARDRAIL_PROVIDER
        self.guardrail_id = guardrail_id
        self.bias_threshold = _thresholds.bias_threshold
        self.hallucination_threshold = _thresholds.hallucination_threshold
        self.risk_flag_threshold = _thresholds.flag_threshold
        self.risk_block_threshold = _thresholds.block_threshold
        self.block_on_high_risk = _behavior.block_on_high_risk
        self.log_only = _behavior.log_only
        self.check_request = _behavior.check_request
        self.check_response = _behavior.check_response
        self.violation_message = _behavior.violation_message
        self.bias_detector = BiasDetector()
        self.hallucination_detector = HallucinationDetector()
        self.risk_scorer = RiskScorer(thresholds=_thresholds, weights=_weights)

    @log_guardrail_information
    async def apply_guardrail(
        self,
        inputs: GenericGuardrailAPIInputs,
        request_data: dict[str, object],
        input_type: Literal["request", "response"],
        logging_obj: LiteLLMLoggingObj | None = None,
    ) -> GenericGuardrailAPIInputs:
        if not self._should_check(input_type):
            return inputs

        start_time = datetime.now(timezone.utc)
        texts = self._extract_texts(inputs)
        if not texts:
            return inputs

        analyses = tuple(self._analyze_text(text=text) for text in texts)
        highest_risk = max(
            analyses, key=lambda analysis: analysis.risk.overall_risk_percentage
        )
        decision = self._decision(highest_risk.risk.recommendation)
        status: GuardrailStatus = (
            "guardrail_intervened" if decision == "blocked" else "success"
        )
        response_payload = self._build_response_payload(
            analyses=analyses,
            decision=decision,
            input_type=input_type,
        )

        self._log_guardrail_result(
            request_data=request_data,
            response_payload=response_payload,
            status=status,
            start_time=start_time,
        )

        if decision == "blocked":
            raise GuardrailRaisedException(
                guardrail_name=self.guardrail_name,
                message=self._build_violation_message(highest_risk.risk),
                should_wrap_with_default_message=False,
            )

        return inputs

    def estimate_bias_hallucination(self, text: str) -> dict[str, object]:
        analysis = self._analyze_text(text=text)
        return {
            "bias": analysis.bias.model_dump(mode="json"),
            "hallucination": analysis.hallucination.model_dump(mode="json"),
            "risk": analysis.risk.model_dump(mode="json"),
        }

    def _should_check(self, input_type: Literal["request", "response"]) -> bool:
        if input_type == "request":
            return self.check_request
        return self.check_response

    def _analyze_text(self, *, text: str) -> TextRiskAnalysis:
        bias_analysis = self.bias_detector.detect(text)
        hallucination_analysis = self.hallucination_detector.detect(text)
        risk = self.risk_scorer.compute_risk(
            bias_analysis=bias_analysis,
            hallucination_analysis=hallucination_analysis,
        )
        return TextRiskAnalysis(
            text=text,
            bias=bias_analysis,
            hallucination=hallucination_analysis,
            risk=risk,
        )

    def _decision(self, recommendation: Literal["pass", "flag", "block"]) -> str:
        if recommendation == "block" and self.block_on_high_risk and not self.log_only:
            return "blocked"
        if recommendation == "pass":
            return "passed"
        return "flagged"

    def _build_response_payload(
        self,
        *,
        analyses: tuple[TextRiskAnalysis, ...],
        decision: str,
        input_type: Literal["request", "response"],
    ) -> dict[str, object]:
        return {
            "decision": decision,
            "input_type": input_type,
            "risk_scores": [
                analysis.risk.model_dump(mode="json") for analysis in analyses
            ],
            "bias": [
                {
                    k: v
                    for k, v in analysis.bias.model_dump(mode="json").items()
                    if k not in _LOG_EXCLUDED_FIELDS
                }
                for analysis in analyses
            ],
            "hallucination": [
                {
                    k: v
                    for k, v in analysis.hallucination.model_dump(mode="json").items()
                    if k not in _LOG_EXCLUDED_FIELDS
                }
                for analysis in analyses
            ],
        }

    def _log_guardrail_result(
        self,
        *,
        request_data: dict[str, object],
        response_payload: dict[str, object],
        status: GuardrailStatus,
        start_time: datetime,
    ) -> None:
        risk_scores = response_payload.get("risk_scores") or []
        highest_risk_percentage: int = max(
            (
                int(r["overall_risk_percentage"])  # type: ignore[index]
                for r in risk_scores
                if isinstance(r, dict) and "overall_risk_percentage" in r
            ),
            default=0,
        )
        detection_methods = self._detection_methods(response_payload)
        tracing_detail = GuardrailTracingDetail(
            guardrail_id=self.guardrail_id or self.guardrail_name,
            detection_method=detection_methods,
            risk_score=highest_risk_percentage / 100,
            violation_categories=self._violation_categories(response_payload),
            guardrail_action=str(response_payload["decision"]),
        )
        self.add_standard_logging_guardrail_information_to_request_data(  # pyright: ignore[reportUnknownMemberType]
            guardrail_provider=self.guardrail_provider,
            guardrail_json_response=response_payload,
            request_data=request_data,
            guardrail_status=status,
            start_time=start_time.timestamp(),
            end_time=datetime.now(timezone.utc).timestamp(),
            duration=(datetime.now(timezone.utc) - start_time).total_seconds(),
            tracing_detail=tracing_detail,
        )

    @staticmethod
    def _detection_methods(response_payload: dict[str, object]) -> str | None:
        categories = BiasHallucinationEstimatorGuardrail._violation_categories(
            response_payload
        )
        if not categories:
            return None
        return "regex,keyword"

    @staticmethod
    def _violation_categories(response_payload: dict[str, object]) -> list[str]:
        risk_scores = response_payload.get("risk_scores")
        if not isinstance(risk_scores, list):
            return []
        typed_risk_scores = cast(list[object], risk_scores)
        return list(
            dict.fromkeys(
                issue.split(":", 1)[0]
                for risk_score in typed_risk_scores
                for issue in BiasHallucinationEstimatorGuardrail._detected_issues(
                    risk_score
                )
            )
        )

    @staticmethod
    def _detected_issues(risk_score: object) -> tuple[str, ...]:
        if not isinstance(risk_score, dict):
            return ()
        typed_risk_score = cast(dict[str, object], risk_score)
        detected_issues = typed_risk_score.get("detected_issues")
        if not isinstance(detected_issues, list):
            return ()
        typed_issues = cast(list[object], detected_issues)
        return tuple(issue for issue in typed_issues if isinstance(issue, str))

    @staticmethod
    def _normalize_event_hook(
        event_hook: GuardrailEventHookInput | None,
    ) -> NormalizedGuardrailEventHook | None:
        if event_hook is None:
            return None
        if isinstance(event_hook, Mode):
            return event_hook
        if isinstance(event_hook, str):
            return GuardrailEventHooks(event_hook)
        return [GuardrailEventHooks(hook) for hook in event_hook]

    def _build_violation_message(self, risk_score: RiskScore) -> str:
        default_message = f"High bias/hallucination risk detected ({risk_score.overall_risk_percentage}%)."
        if self.violation_message:
            return self.violation_message
        return self.render_violation_message(
            default=default_message,
            context={
                "risk_score": risk_score.overall_risk_percentage,
                "detected_issues": ", ".join(risk_score.detected_issues),
            },
        )

    @staticmethod
    def _extract_texts(inputs: GenericGuardrailAPIInputs) -> tuple[str, ...]:
        texts = tuple(text for text in inputs.get("texts", []) if text)
        tool_call_texts = tuple(
            text
            for tool_call in inputs.get("tool_calls", [])
            for text in (
                BiasHallucinationEstimatorGuardrail._tool_call_text(tool_call),
            )
            if text
        )
        return texts + tool_call_texts

    @staticmethod
    def _tool_call_text(tool_call: object) -> str | None:
        if isinstance(tool_call, dict):
            tool_call_dict = cast(Mapping[str, object], tool_call)
            function = tool_call_dict.get("function")
            if isinstance(function, dict):
                function_dict = cast(Mapping[str, object], function)
                name = function_dict.get("name")
                arguments = function_dict.get("arguments")
                return (
                    f"{BiasHallucinationEstimatorGuardrail._string_value(name)} "
                    f"{BiasHallucinationEstimatorGuardrail._string_value(arguments)}"
                ).strip() or None
            return None

        if not hasattr(tool_call, "function"):
            return None
        tool_call_like = cast(ToolCallLike, tool_call)
        function = tool_call_like.function
        name = function.name or ""
        arguments = function.arguments
        return f"{name} {arguments}".strip() or None

    @staticmethod
    def _string_value(value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def get_config_model() -> type[GuardrailConfigModel[BaseModel]] | None:
        from litellm.types.proxy.guardrails.guardrail_hooks.bias_hallucination_estimator import (
            BiasHallucinationEstimatorConfigModel,
        )

        return cast(
            type[GuardrailConfigModel[BaseModel]],
            BiasHallucinationEstimatorConfigModel,
        )


BiasHallucinationEstimator = BiasHallucinationEstimatorGuardrail
