from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Final, Literal
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import litellm
from litellm.exceptions import BudgetExceededError
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.common_utils.rerank_shortlist import (
    RerankFailed,
    RerankScore,
    rerank_order,
    router_reranker,
)
from litellm.types.rerank import RerankResponse

CALLER: Final = UserAPIKeyAuth(api_key="hashed-caller-key", team_id="team-1", user_id="user-1")
QUERY: Final = "convert dollars to euros"
DOCUMENTS: Final = ("weather-forecast", "treasury-get_rates", "calendar-create_event")


class FakeReranker:
    def __init__(self, scores: Sequence[RerankScore] = (), error: Exception | None = None) -> None:
        self.scores: Final = tuple(scores)
        self.error: Final = error
        self.calls: list[tuple[str, tuple[str, ...]]] = []  # mutable-ok: test spy recording rerank inputs

    async def __call__(self, query: str, documents: Sequence[str]) -> Sequence[RerankScore]:
        self.calls.append((query, tuple(documents)))
        if self.error is not None:
            raise self.error
        return self.scores


def _pass_through_key_limits() -> MagicMock:
    limits = MagicMock()
    limits.pre_call_hook = AsyncMock(side_effect=lambda user_api_key_dict, data, call_type: data)
    return limits


def _rerank_router(scores: Sequence[float] = (0.1, 0.9, 0.4)) -> MagicMock:
    router = MagicMock()
    router.arerank = AsyncMock(
        side_effect=lambda model, query, documents, top_n, metadata, **_: RerankResponse(
            id="rerank-1",
            results=[{"index": index, "relevance_score": score} for index, score in enumerate(scores)],
        )
    )
    return router


@pytest.mark.asyncio
async def test_scores_come_back_best_first_with_ties_in_document_order() -> None:
    reranker = FakeReranker(scores=(RerankScore(0, 0.5), RerankScore(1, 0.9), RerankScore(2, 0.5)))

    assert await rerank_order(QUERY, DOCUMENTS, reranker) == (
        RerankScore(1, 0.9),
        RerankScore(0, 0.5),
        RerankScore(2, 0.5),
    )
    assert reranker.calls == [(QUERY, DOCUMENTS)]


@pytest.mark.asyncio
async def test_scores_outside_zero_to_one_are_accepted() -> None:
    reranker = FakeReranker(scores=(RerankScore(0, -3.5), RerankScore(1, 7.25), RerankScore(2, 0.0)))

    assert await rerank_order(QUERY, DOCUMENTS, reranker) == (
        RerankScore(1, 7.25),
        RerankScore(2, 0.0),
        RerankScore(0, -3.5),
    )


@pytest.mark.parametrize(
    ("scores", "reason"),
    (
        (
            (RerankScore(0, 0.9), RerankScore(0, 0.8), RerankScore(2, 0.1)),
            "rerank model did not score each of the 3 documents exactly once",
        ),
        (
            (RerankScore(0, 0.9), RerankScore(1, 0.8), RerankScore(3, 0.1)),
            "rerank model did not score each of the 3 documents exactly once",
        ),
        (
            (RerankScore(-1, 0.9), RerankScore(1, 0.8), RerankScore(2, 0.1)),
            "rerank model did not score each of the 3 documents exactly once",
        ),
        (
            (RerankScore(0, 0.9), RerankScore(1, 0.8)),
            "rerank model did not score each of the 3 documents exactly once",
        ),
        (
            (RerankScore(0, 0.9), RerankScore(1, 0.8), RerankScore(2, 0.1), RerankScore(2, 0.1)),
            "rerank model did not score each of the 3 documents exactly once",
        ),
        (
            (RerankScore(0, math.nan), RerankScore(1, 0.8), RerankScore(2, 0.1)),
            "rerank model returned a score that is not a finite number",
        ),
        (
            (RerankScore(0, 0.9), RerankScore(1, math.inf), RerankScore(2, 0.1)),
            "rerank model returned a score that is not a finite number",
        ),
    ),
    ids=("duplicate-index", "index-equal-to-length", "negative-index", "missing", "extra", "nan", "infinity"),
)
@pytest.mark.asyncio
async def test_an_invalid_answer_is_a_failure(scores: tuple[RerankScore, ...], reason: str) -> None:
    assert await rerank_order(QUERY, DOCUMENTS, FakeReranker(scores=scores)) == RerankFailed(reason=reason)


@pytest.mark.parametrize(
    "error",
    (
        litellm.Timeout(message="rerank timed out", model="rerank-2.5", llm_provider="voyage"),
        litellm.APIConnectionError(message="provider unreachable", model="rerank-2.5", llm_provider="voyage"),
        BudgetExceededError(current_cost=2.0, max_budget=1.0, message="budget exceeded"),
        ValueError("response body did not parse"),
    ),
    ids=("timeout", "provider-error", "budget", "unparseable-body"),
)
@pytest.mark.asyncio
async def test_a_failed_call_is_a_failure(error: Exception) -> None:
    outcome: Final = await rerank_order(QUERY, DOCUMENTS, FakeReranker(error=error))

    assert outcome == RerankFailed(reason=f"rerank call failed: {error}")


@pytest.mark.parametrize(
    "error",
    (ProxyRateLimitError(detail="rpm exceeded"), HTTPException(status_code=400, detail="blocked by guardrail")),
    ids=("rate-limit", "guardrail"),
)
@pytest.mark.asyncio
async def test_admission_refusals_reach_the_caller(error: HTTPException) -> None:
    with pytest.raises(HTTPException) as raised:
        await rerank_order(QUERY, DOCUMENTS, FakeReranker(error=error))

    assert raised.value is error


@pytest.mark.asyncio
async def test_router_reranker_asks_for_every_document_and_attributes_spend_to_the_caller() -> None:
    router = _rerank_router()
    key_limits = _pass_through_key_limits()

    outcome: Final = await rerank_order(QUERY, DOCUMENTS, router_reranker(router, "rerank-2.5", CALLER, key_limits))

    assert outcome == (RerankScore(1, 0.9), RerankScore(2, 0.4), RerankScore(0, 0.1))
    sent: Final = router.arerank.await_args.kwargs
    assert sent["model"] == "rerank-2.5"
    assert sent["query"] == QUERY
    assert sent["documents"] == list(DOCUMENTS)
    assert sent["top_n"] == 3
    assert sent["metadata"]["user_api_key"] == "hashed-caller-key"
    assert sent["metadata"]["user_api_key_team_id"] == "team-1"
    assert sent["metadata"]["user_api_key_user_id"] == "user-1"
    checked: Final = key_limits.pre_call_hook.await_args.kwargs
    assert checked["user_api_key_dict"] is CALLER
    assert checked["call_type"] == "arerank"
    assert checked["data"]["model"] == "rerank-2.5"
    assert checked["data"]["query"] == QUERY
    assert checked["data"]["documents"] == list(DOCUMENTS)


@pytest.mark.parametrize(
    ("region", "forwarded"),
    (("eu", {"allowed_model_region": "eu"}), (None, {})),
    ids=("region-restricted-caller", "unrestricted-caller"),
)
@pytest.mark.asyncio
async def test_the_callers_model_region_reaches_the_router_like_on_rerank(
    region: Literal["eu"] | None, forwarded: Mapping[str, str]
) -> None:
    router = _rerank_router()
    caller: Final = UserAPIKeyAuth(api_key="hashed-caller-key", allowed_model_region=region)

    await rerank_order(QUERY, DOCUMENTS, router_reranker(router, "rerank-2.5", caller, _pass_through_key_limits()))

    sent: Final = router.arerank.await_args.kwargs
    assert {key: sent[key] for key in sent if key == "allowed_model_region"} == forwarded


@pytest.mark.asyncio
async def test_the_rerank_model_sees_the_query_as_the_guardrails_rewrote_it() -> None:
    router = _rerank_router()
    key_limits = MagicMock()
    key_limits.pre_call_hook = AsyncMock(
        side_effect=lambda user_api_key_dict, data, call_type: {
            **data,
            "query": "[MASKED]",
            "metadata": {**data["metadata"], "guardrail": "masked"},
        }
    )

    await rerank_order(QUERY, DOCUMENTS, router_reranker(router, "rerank-2.5", CALLER, key_limits))

    sent: Final = router.arerank.await_args.kwargs
    assert sent["query"] == "[MASKED]"
    assert sent["metadata"]["guardrail"] == "masked"


def _pii_guardrail(
    user_api_key_dict: UserAPIKeyAuth, data: Mapping[str, Mapping[str, object]], call_type: str
) -> Mapping[str, Mapping[str, object]]:
    assigned: Final = data["metadata"].get("guardrails")
    if isinstance(assigned, list) and "pii-mask" in assigned:
        raise HTTPException(status_code=400, detail="blocked by pii-mask")
    return data


@pytest.mark.asyncio
async def test_a_guardrail_assigned_to_the_calling_key_runs_on_the_rerank_call() -> None:
    caller: Final = UserAPIKeyAuth(api_key="hashed-caller-key", metadata={"guardrails": ["pii-mask"]})
    router = _rerank_router()
    key_limits = MagicMock()
    key_limits.pre_call_hook = AsyncMock(side_effect=_pii_guardrail)

    with (
        patch(  # test-quality-ok: key guardrails are gated on this proxy global, as on /rerank
            "litellm.proxy.proxy_server.premium_user", True
        ),
        pytest.raises(HTTPException) as raised,
    ):
        await rerank_order(QUERY, DOCUMENTS, router_reranker(router, "rerank-2.5", caller, key_limits))

    assert raised.value.detail == "blocked by pii-mask"
    router.arerank.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_key_over_its_limit_never_reaches_the_rerank_model() -> None:
    router = _rerank_router()
    key_limits = MagicMock()
    key_limits.pre_call_hook = AsyncMock(side_effect=ProxyRateLimitError(detail="rpm exceeded"))

    with pytest.raises(ProxyRateLimitError) as raised:
        await rerank_order(QUERY, DOCUMENTS, router_reranker(router, "rerank-2.5", CALLER, key_limits))

    assert raised.value.status_code == 429
    router.arerank.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_response_without_results_is_a_failure() -> None:
    router = MagicMock()
    router.arerank = AsyncMock(return_value=RerankResponse(id="rerank-1", results=None))

    outcome: Final = await rerank_order(
        QUERY, DOCUMENTS, router_reranker(router, "rerank-2.5", CALLER, _pass_through_key_limits())
    )

    assert isinstance(outcome, RerankFailed)
    assert outcome.reason.startswith("rerank call failed: 1 validation error for _RerankData")
