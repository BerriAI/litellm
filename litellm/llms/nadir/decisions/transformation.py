import json
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError, model_validator
from pydantic_core import to_jsonable_python
from typing_extensions import assert_never

from litellm.constants import NADIR_DEFAULT_API_BASE
from litellm.llms.base_llm.decisions.transformation import (
    BaseDecisionsConfig,
    parse_systemone_response,
    systemone_keys,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.decisions import (
    DecisionsIRChoiceQuestion,
    DecisionsIRMessages,
    DecisionsIRRequest,
    DecisionsIRResponse,
    DecisionsIRState,
    DecisionsJSON,
    OpenAIDecisionContentPart,
    OpenAIDecisionInputText,
    UnsupportedDecisionsRequest,
)
from litellm.types.llms.base import LiteLLMBaseModel

NADIR_CLASSIFIER_MODEL: Final = "bucket"
_REQUEST_SOURCE: Final = "litellm"

NadirTier: TypeAlias = Literal["simple", "medium", "complex"]
_TIERS: Final[tuple[NadirTier, ...]] = ("simple", "medium", "complex")
_CANONICAL_CHOICES: Final[Mapping[NadirTier, str]] = MappingProxyType({tier: tier.upper() for tier in _TIERS})
_Probability: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]
_MAPPING_ADAPTER: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])
_UNSUPPORTED_QUESTIONS: Final = UnsupportedDecisionsRequest(
    reason="Nadir answers exactly one choice question whose choices include simple, medium and complex"
)


class _NadirBucketVerdict(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    routing_tier: NadirTier
    confidence: _Probability | None = None
    probabilities: Mapping[str, _Probability] = Field(default_factory=lambda: MappingProxyType({}))

    @model_validator(mode="before")
    @classmethod
    def _routing_tier_falls_back_to_bucket(cls, data: object) -> object:
        """``routing_tier`` is the tier Nadir itself routes on (the bucket after its confidence and gate
        adjustments), so it wins; an older server that sends only ``bucket`` still answers."""
        try:
            fields: Final = _MAPPING_ADAPTER.validate_python(data)
        except ValidationError:
            return data
        return {**fields, "routing_tier": fields.get("routing_tier") or fields.get("bucket")}


_VERDICT_ADAPTER: Final = TypeAdapter(_NadirBucketVerdict)


class _ConversationMessage(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant"]
    content: str


class _ConversationState(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    messages: tuple[_ConversationMessage, ...] = Field(min_length=1)


_CONVERSATION_ADAPTER: Final = TypeAdapter(_ConversationState)


def _part_text(content: str | Sequence[OpenAIDecisionContentPart]) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(part.text for part in content if isinstance(part, OpenAIDecisionInputText))


def _state_body(state: DecisionsJSON) -> Mapping[str, object]:
    if isinstance(state, str):
        return {"prompt": state}
    try:
        conversation: Final = _CONVERSATION_ADAPTER.validate_python(state)
    except ValidationError:
        return {"prompt": json.dumps(to_jsonable_python(state))}
    return {"messages": [message.model_dump() for message in conversation.messages]}


def _input_body(decision_input: DecisionsIRState | DecisionsIRMessages) -> Mapping[str, object]:
    match decision_input:
        case DecisionsIRMessages(messages=messages):
            return {
                "messages": [{"role": message.role, "content": _part_text(message.content)} for message in messages]
            }
        case DecisionsIRState(state=state):
            return _state_body(state)
        case _:
            assert_never(decision_input)


def nadir_bucket_body(decision_input: DecisionsIRState | DecisionsIRMessages) -> Mapping[str, object]:
    """The ``/v1/bucket`` body for a decision input.

    Nadir's classifier is trained on raw conversations, so a state shaped ``{"messages": [{role, content}]}``
    goes out as those messages, as do OpenAI-format input messages; any other state is the prompt. Plain
    dicts and lists, because httpx serializes ``json=`` with json.dumps, which rejects MappingProxyType.
    """
    return {**_input_body(decision_input), "source": _REQUEST_SOURCE}


def _tier_choices(request: DecisionsIRRequest) -> Mapping[NadirTier, str] | None:
    """The question's own choice key for each Nadir tier, matched case-insensitively, when the request is
    one choice question naming all three tiers."""
    match request.questions:
        case (DecisionsIRChoiceQuestion(choices=choices),):
            by_name: Final = {option.value.lower(): option.value for option in choices if isinstance(option.value, str)}
            by_tier: Final[Mapping[NadirTier, str]] = {tier: by_name[tier] for tier in _TIERS if tier in by_name}
            return MappingProxyType(by_tier) if len(by_tier) == len(_TIERS) else None
        case _:
            return None


def _choice_answer(verdict: _NadirBucketVerdict, choices: Mapping[NadirTier, str]) -> Mapping[str, object]:
    tier: Final = verdict.routing_tier
    return {
        "type": "choice",
        "choice": choices[tier],
        "probabilities": {choices[name]: verdict.probabilities.get(name, 0.0) for name in _TIERS},
        "confidence": verdict.confidence if verdict.confidence is not None else verdict.probabilities.get(tier, 0.0),
    }


class NadirDecisionsConfig(BaseDecisionsConfig):
    """Nadir as a Decisions provider: ``/v1/bucket`` answers a single simple/medium/complex choice question.

    Nadir is a tier classifier, not a general judge, so any other question set is rejected before a call.
    ``NADIR_API_KEY`` is only sent to ``NADIR_API_BASE`` or the hosted API, the same scoping as the ``nadir``
    chat provider; with no key the call is anonymous, which Nadir serves on a per-IP rate-limited tier.
    """

    path = "/v1/bucket"
    api_key_env = ("NADIR_API_KEY",)
    api_base_env = ("NADIR_API_BASE",)
    api_key_required = False
    health_check_questions = MappingProxyType(
        {
            "tier": MappingProxyType(
                {
                    "type": "choice",
                    "instructions": "Which tier should serve this request?",
                    "criteria": {tier: None for tier in _TIERS},
                }
            )
        }
    )

    def get_default_api_base(self) -> str:
        return NADIR_DEFAULT_API_BASE

    def resolve_connection(self, api_base: str | None, api_key: str | None) -> tuple[str, str | None]:
        default_base: Final = get_secret_str("NADIR_API_BASE") or NADIR_DEFAULT_API_BASE
        trusted_base: Final = api_base is None or api_base.rstrip("/") == default_base.rstrip("/")
        return (
            (api_base or default_base).rstrip("/"),
            api_key or (get_secret_str("NADIR_API_KEY") if trusted_base else None),
        )

    def transform_decisions_request(
        self,
        model: str,
        request: DecisionsIRRequest,
        custom_llm_provider: str,
    ) -> Mapping[str, object] | UnsupportedDecisionsRequest:
        if _tier_choices(request) is None:
            return _UNSUPPORTED_QUESTIONS
        return nadir_bucket_body(request.input)

    def parse_response(self, payload: object, request: DecisionsIRRequest) -> DecisionsIRResponse:
        verdict: Final = _VERDICT_ADAPTER.validate_python(payload)
        choices: Final = _tier_choices(request)
        answers: Final = (
            {}
            if choices is None
            else {key: _choice_answer(verdict, choices) for key in systemone_keys(request.questions)}
        )
        return parse_systemone_response({"answers": answers}, request)

    def classifier_response(self, body: Mapping[str, object], requested_model: str) -> Mapping[str, object]:
        """The verdict as the System One ``tier`` answer the auto router reads, on canonical tier labels the
        router resolves whatever display labels it uses."""
        verdict: Final = _VERDICT_ADAPTER.validate_python(body)
        return MappingProxyType(
            {"model": requested_model, "answers": {"tier": _choice_answer(verdict, _CANONICAL_CHOICES)}}
        )


NADIR_DECISIONS_CONFIG: Final[NadirDecisionsConfig] = NadirDecisionsConfig()
