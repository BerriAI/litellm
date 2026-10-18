from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import ConfigDict, Field, TypeAdapter

from litellm.constants import NADIR_DEFAULT_API_BASE
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.base import LiteLLMBaseModel

NADIR_CLASSIFIER_MODEL: Final = "bucket"
_BUCKET_PATH: Final = "/v1/bucket"
_REQUEST_SOURCE: Final = "litellm"

NadirTier: TypeAlias = Literal["simple", "medium", "complex"]
_Probability: TypeAlias = Annotated[float, Field(ge=0.0, le=1.0)]


class _NadirBucketVerdict(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    routing_tier: NadirTier | None = None
    bucket: NadirTier | None = None
    confidence: _Probability | None = None
    probabilities: Mapping[str, _Probability] = Field(default_factory=lambda: MappingProxyType({}))


_VERDICT_ADAPTER: Final = TypeAdapter(_NadirBucketVerdict)


@dataclass(frozen=True, slots=True)
class NadirClassifierConnection:
    api_base: str
    api_key: str | None = field(repr=False)


def nadir_classifier_connection(api_base: str | None, api_key: str | None) -> NadirClassifierConnection:
    """Resolve where the classifier calls go, with the chat provider's key scoping.

    ``NADIR_API_KEY`` is only sent to ``NADIR_API_BASE`` or the hosted API. With no key at all the
    call is anonymous, which Nadir serves on a per-IP rate-limited tier.
    """
    default_base: Final = get_secret_str("NADIR_API_BASE") or NADIR_DEFAULT_API_BASE
    trusted_base: Final = api_base is None or api_base.rstrip("/") == default_base.rstrip("/")
    return NadirClassifierConnection(
        api_base=(api_base or default_base).rstrip("/"),
        api_key=api_key or (get_secret_str("NADIR_API_KEY") if trusted_base else None),
    )


def nadir_bucket_url(api_base: str) -> str:
    # Nadir documents its base with the /v1 suffix for OpenAI-compatible clients.
    return f"{api_base.rstrip('/').removesuffix('/v1')}{_BUCKET_PATH}"


def nadir_bucket_request(state: str, messages: Sequence[Mapping[str, str]] | None = None) -> Mapping[str, object]:
    """Nadir's classifier reads the conversation itself, so it gets the messages rather than a judge prompt.

    Plain dicts, because httpx serializes ``json=`` with json.dumps, which rejects MappingProxyType.
    """
    if messages:
        return {"messages": [dict(message) for message in messages], "source": _REQUEST_SOURCE}
    return {"prompt": state, "source": _REQUEST_SOURCE}


def nadir_classifier_response(body: Mapping[str, object], requested_model: str) -> Mapping[str, object]:
    """Translate a ``/v1/bucket`` verdict into the System One ``tier`` choice answer the router reads.

    ``routing_tier`` is the tier Nadir itself routes on (the bucket after its confidence and gate
    adjustments), so it wins over ``bucket``. Labels are the canonical tier names, which the router
    resolves whatever display labels it uses.
    """
    verdict: Final = _VERDICT_ADAPTER.validate_python(body)
    tier: Final = verdict.routing_tier or verdict.bucket
    if tier is None:
        raise ValueError("Nadir response names neither routing_tier nor bucket")
    confidence: Final = verdict.confidence if verdict.confidence is not None else verdict.probabilities.get(tier, 0.0)
    answer: Final = MappingProxyType(
        {
            "type": "choice",
            "choice": tier.upper(),
            "probabilities": MappingProxyType({name.upper(): p for name, p in verdict.probabilities.items()}),
            "confidence": confidence,
        }
    )
    return MappingProxyType({"model": requested_model, "answers": MappingProxyType({"tier": answer})})
