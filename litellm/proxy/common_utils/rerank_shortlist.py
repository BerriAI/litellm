"""Re-orders a search shortlist with a rerank model from model_list, keeping the caller's order when it fails."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from fastapi import HTTPException
from openai import OpenAIError
from pydantic import ConfigDict

from litellm.exceptions import BudgetExceededError
from litellm.proxy.common_utils.semantic_text_index import embedding_spend_metadata
from litellm.types.llms.base import LiteLLMBaseModel

if TYPE_CHECKING:
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.utils import ProxyLogging
    from litellm.router import Router

RERANK_SHORTLIST_SIZE: Final = 50
"""How many base-ranked candidates one rerank call sees, so it can promote ones below the requested top_k."""


@dataclass(frozen=True, slots=True)
class RerankScore:
    index: int
    score: float


@dataclass(frozen=True, slots=True)
class RerankFailed:
    reason: str


class Reranker(Protocol):
    def __call__(self, query: str, documents: Sequence[str]) -> Awaitable[Sequence[RerankScore]]: ...


class _RerankResult(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)

    index: int
    relevance_score: float


class _RerankData(LiteLLMBaseModel):
    model_config = ConfigDict(frozen=True, extra="ignore", from_attributes=True)

    results: tuple[_RerankResult, ...]


class _RerankRequest(LiteLLMBaseModel):
    """The /rerank-shaped request as the pre-call hooks (rate limits, budgets, guardrails) hand it back."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    model: str
    query: str
    documents: tuple[str, ...]
    metadata: dict[str, object]  # mutable-ok: the router mutates the metadata dict it is handed


def router_reranker(
    router: Router, rerank_model: str, user_api_key_dict: UserAPIKeyAuth, proxy_logging_obj: ProxyLogging
) -> Reranker:
    """Reranks through the router after the same key rate-limit, budget and guardrail pre-call hooks /rerank runs."""

    async def rerank(query: str, documents: Sequence[str]) -> Sequence[RerankScore]:
        from litellm.proxy.litellm_pre_call_utils import (
            move_guardrails_to_metadata,  # pyright: ignore[reportUnknownVariableType]  # it takes an untyped request dict
        )

        request: Final = {
            "model": rerank_model,
            "query": query,
            "documents": list(documents),
            "metadata": embedding_spend_metadata(user_api_key_dict),
        }
        await move_guardrails_to_metadata(
            data=request, _metadata_variable_name="metadata", user_api_key_dict=user_api_key_dict
        )
        processed: Final = _RerankRequest.model_validate(
            await proxy_logging_obj.pre_call_hook(  # pyright: ignore[reportUnknownMemberType]  # its overloads take an untyped dict
                user_api_key_dict=user_api_key_dict, data=request, call_type="arerank"
            )
        )
        region: Final = user_api_key_dict.allowed_model_region
        response: Final = await router.arerank(
            model=processed.model,
            query=processed.query,
            documents=list(processed.documents),
            top_n=len(processed.documents),
            metadata=processed.metadata,
            **({} if region is None else {"allowed_model_region": region}),
        )
        return tuple(
            RerankScore(index=result.index, score=result.relevance_score)
            for result in _RerankData.model_validate(response).results
        )

    return rerank


def _invalid_answer(scores: Sequence[RerankScore], document_count: int) -> str | None:
    if len(scores) != document_count or frozenset(score.index for score in scores) != frozenset(range(document_count)):
        return f"rerank model did not score each of the {document_count} documents exactly once"
    if not all(math.isfinite(score.score) for score in scores):
        return "rerank model returned a score that is not a finite number"
    return None


async def rerank_order(
    query: str, documents: Sequence[str], rerank: Reranker
) -> tuple[RerankScore, ...] | RerankFailed:
    """Scores best first, ties in document order. A rate-limit or guardrail HTTPException still reaches the caller."""
    try:
        scores: Final = tuple(await rerank(query, documents))
    except HTTPException:
        raise
    except (OpenAIError, ValueError, BudgetExceededError) as exc:
        return RerankFailed(reason=f"rerank call failed: {exc}")
    invalid: Final = _invalid_answer(scores, len(documents))
    if invalid is not None:
        return RerankFailed(reason=invalid)
    return tuple(sorted(scores, key=lambda score: (-score.score, score.index)))
