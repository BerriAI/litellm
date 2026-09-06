"""
Translate between Cohere's `/rerank` format and Cloudflare Workers AI's `/ai/run/<model>` format.

Workers AI reranking is not served by the OpenAI-compatible `/ai/v1` surface, so this targets
the native run path. Its request uses `contexts[].text` instead of `documents[]`, and its
response returns `{id, score}` pairs where `score` is a raw logit rather than a 0-1 relevance.
"""

import math
from collections.abc import Mapping, Sequence
from typing import Final

import httpx
from pydantic import BaseModel, ConfigDict, ValidationError

from litellm._uuid import uuid
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.rerank.transformation import BaseRerankConfig
from litellm.secret_managers.main import get_secret_str, normalize_nonempty_secret_str
from litellm.types.rerank import (
    OptionalRerankParams,
    RerankBilledUnits,
    RerankResponse,
    RerankResponseDocument,
    RerankResponseMeta,
    RerankResponseResult,
    RerankTokens,
)

from ..common_utils import CloudflareError

RUN_PATH: Final = "/ai/run"
OPENAI_COMPAT_PATH: Final = "/ai/v1"
SUPPORTED_COHERE_PARAMS: Final = ("query", "documents", "top_n", "return_documents")


class CloudflareRerankParams(BaseModel):
    """The Cohere-format rerank params this provider needs, validated off the untyped param dict."""

    model_config = ConfigDict(frozen=True)

    query: str
    documents: tuple[str | Mapping[str, object], ...]
    top_n: int | None = None


class CloudflareRerankContext(BaseModel):
    """One entry of the `contexts` array Workers AI scores against."""

    model_config = ConfigDict(frozen=True)

    text: str


class CloudflareRerankRequest(BaseModel):
    """The body Workers AI reranking expects at `/ai/run/<model>`."""

    model_config = ConfigDict(frozen=True)

    query: str
    contexts: tuple[CloudflareRerankContext, ...]
    top_k: int | None = None


class CloudflareRerankScore(BaseModel):
    """One `{id, score}` pair, where `id` indexes into the request's `contexts`."""

    model_config = ConfigDict(frozen=True)

    id: int
    score: float


class CloudflareRerankResult(BaseModel):
    """The `result` block of a Workers AI reranking response."""

    model_config = ConfigDict(frozen=True)

    response: tuple[CloudflareRerankScore, ...] | None = None


class CloudflareRerankEnvelope(BaseModel):
    """The Cloudflare v4 API envelope wrapping a Workers AI reranking result."""

    model_config = ConfigDict(frozen=True)

    result: CloudflareRerankResult | None = None
    success: bool = True
    errors: tuple[Mapping[str, object], ...] = ()


def sigmoid(score: float) -> float:
    """Map a raw reranker logit onto the 0-1 relevance range Cohere clients expect."""
    if score >= 0:
        return 1.0 / (1.0 + math.exp(-score))
    exponentiated: Final = math.exp(score)
    return exponentiated / (1.0 + exponentiated)


def context_text(document: str | Mapping[str, object]) -> str:
    """Read the scorable text out of one Cohere-format rerank document."""
    if isinstance(document, str):
        return document
    text: Final = document.get("text")
    if isinstance(text, str):
        return text
    raise ValueError(f"Cloudflare rerank documents must be strings or objects with a 'text' field, got {document!r}")


def validated_rerank_params(optional_rerank_params: Mapping[str, object]) -> CloudflareRerankParams:
    """Turn the untyped Cohere param mapping into the typed params this provider sends."""
    try:
        return CloudflareRerankParams.model_validate(optional_rerank_params)
    except ValidationError as error:
        raise ValueError(f"Invalid Cloudflare rerank request: {error}") from error


def validated_rerank_envelope(raw_response: httpx.Response) -> CloudflareRerankEnvelope:
    """Decode a Workers AI reranking response, or fail with the body that could not be read."""
    try:
        return CloudflareRerankEnvelope.model_validate_json(raw_response.text)
    except ValidationError as error:
        raise CloudflareError(
            status_code=raw_response.status_code,
            message=f"Error parsing Cloudflare rerank response: {raw_response.text}",
        ) from error


def context_texts(request_data: Mapping[str, object] | None) -> tuple[str, ...]:
    """Recover the request's context texts so scored indices can be echoed back as documents."""
    if request_data is None:
        return ()
    try:
        request: Final = CloudflareRerankRequest.model_validate(request_data)
    except ValidationError:
        return ()
    return tuple(context.text for context in request.contexts)


class CloudflareRerankConfig(BaseRerankConfig):
    def get_complete_url(
        self,
        api_base: str | None,
        model: str,
        optional_params: Mapping[str, object] | None = None,
    ) -> str:
        if api_base is None:
            account_id: Final = normalize_nonempty_secret_str(get_secret_str("CLOUDFLARE_ACCOUNT_ID"))
            if account_id is None:
                raise ValueError(
                    "Missing CLOUDFLARE_ACCOUNT_ID - set CLOUDFLARE_ACCOUNT_ID in the environment "
                    "or pass api_base explicitly"
                )
            return f"https://api.cloudflare.com/client/v4/accounts/{account_id}{RUN_PATH}/{model}"

        trimmed: Final = api_base.rstrip("/")
        if trimmed.endswith(f"{RUN_PATH}/{model}"):
            return trimmed
        if trimmed.endswith(OPENAI_COMPAT_PATH):
            return f"{trimmed[: -len(OPENAI_COMPAT_PATH)]}{RUN_PATH}/{model}"
        if trimmed.endswith(RUN_PATH):
            return f"{trimmed}/{model}"
        return f"{trimmed}{RUN_PATH}/{model}"

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        optional_params: Mapping[str, object] | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict:
        resolved_key: Final = api_key or get_secret_str("CLOUDFLARE_API_KEY")
        if resolved_key is None:
            raise ValueError(
                "Missing Cloudflare API key - set CLOUDFLARE_API_KEY in the environment or pass api_key explicitly"
            )
        defaults: Final = (
            ("Authorization", f"Bearer {resolved_key}"),
            ("accept", "application/json"),
            ("content-type", "application/json"),
        )
        return dict((*defaults, *headers.items()))  # mutable-ok: BaseRerankConfig fixes this return type as dict

    def get_supported_cohere_rerank_params(self, model: str) -> list:
        return sorted(SUPPORTED_COHERE_PARAMS)

    def map_cohere_rerank_params(
        self,
        non_default_params: Mapping[str, object] | None,
        model: str,
        drop_params: bool,
        query: str,
        documents: list[str | dict[str, object]],
        custom_llm_provider: str | None = None,
        top_n: int | None = None,
        rank_fields: Sequence[str] | None = None,
        return_documents: bool | None = True,
        max_chunks_per_doc: int | None = None,
        max_tokens_per_doc: int | None = None,
        instruction: str | None = None,
    ) -> dict:
        rejected: Final = tuple(
            name
            for name, value in (
                ("rank_fields", rank_fields),
                ("max_chunks_per_doc", max_chunks_per_doc),
                ("max_tokens_per_doc", max_tokens_per_doc),
                ("instruction", instruction),
            )
            if value is not None
        )
        if rejected and not drop_params:
            raise UnsupportedParamsError(
                status_code=400,
                message=f"cloudflare rerank does not support {', '.join(rejected)}. Pass `drop_params=True` to ignore.",
            )
        return OptionalRerankParams(query=query, documents=documents, top_n=top_n)

    def transform_rerank_request(
        self,
        model: str,
        optional_rerank_params: Mapping[str, object],
        headers: Mapping[str, str],
        litellm_params: Mapping[str, object] | None = None,
    ) -> dict:
        params: Final = validated_rerank_params(optional_rerank_params)
        return CloudflareRerankRequest(
            query=params.query,
            contexts=tuple(CloudflareRerankContext(text=context_text(document)) for document in params.documents),
            top_k=params.top_n,
        ).model_dump(exclude_none=True, mode="json")

    def transform_rerank_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: RerankResponse,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None = None,
        request_data: Mapping[str, object] | None = None,
        optional_params: Mapping[str, object] | None = None,
        litellm_params: Mapping[str, object] | None = None,
    ) -> RerankResponse:
        logging_obj.post_call(original_response=raw_response.text)

        envelope: Final = validated_rerank_envelope(raw_response)
        if not envelope.success:
            raise CloudflareError(
                status_code=raw_response.status_code,
                message=f"Cloudflare rerank request failed: {envelope.errors}",
            )
        scores: Final = envelope.result.response if envelope.result is not None else None
        if scores is None:
            raise CloudflareError(
                status_code=raw_response.status_code,
                message=f"No rerank results found in the Cloudflare response={raw_response.text}",
            )

        contexts: Final = context_texts(request_data)
        return RerankResponse(
            id=str(uuid.uuid4()),
            results=tuple(self._transform_score(score, contexts) for score in scores),
            meta=RerankResponseMeta(billed_units=RerankBilledUnits(), tokens=RerankTokens()),
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        return CloudflareError(status_code=status_code, message=error_message)

    @staticmethod
    def _transform_score(score: CloudflareRerankScore, contexts: Sequence[str]) -> RerankResponseResult:
        scored: Final = RerankResponseResult(index=score.id, relevance_score=sigmoid(score.score))
        if not 0 <= score.id < len(contexts):
            return scored
        with_document: Final[RerankResponseResult] = {
            **scored,
            "document": RerankResponseDocument(text=contexts[score.id]),
        }
        return with_document
