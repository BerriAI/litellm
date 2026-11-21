from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.routing_common_utils import resolve_model_group_alias
from litellm.proxy._types import UserAPIKeyAuth
from litellm.router import Router

from .benchmarks import EPOCH_URL, BenchmarkCache
from .models import Candidate, DataSource, FrozenModel, RecommendationRequest, RecommendationResponse
from .scoring import matches, rank
from .speed import QueryClient, SpeedCache

BENCHMARK_CACHE: Final = BenchmarkCache()
SPEED_CACHE: Final = SpeedCache()


class DeploymentParams(FrozenModel):
    model: str | None = None


class Deployment(FrozenModel):
    litellm_params: DeploymentParams


def candidate(row: object, deployments: object) -> Candidate:
    base: Final = Candidate.model_validate(row, from_attributes=True)
    parsed: Final = TypeAdapter(tuple[Deployment, ...]).validate_python(deployments or ())
    return base.model_copy(
        update={
            "underlying_models": tuple(
                deployment.litellm_params.model for deployment in parsed if deployment.litellm_params.model is not None
            )
        }
    )


async def recommend_models(
    request: RecommendationRequest,
    auth: UserAPIKeyAuth,
    listing: object,
    llm_router: Router | None,
    lookup: Mapping[str, str],
    client: QueryClient | None,
    cache: BenchmarkCache = BENCHMARK_CACHE,
    speed_cache: SpeedCache = SPEED_CACHE,
) -> RecommendationResponse:
    rows: Final = TypeAdapter(Mapping[str, tuple[object, ...]]).validate_python(listing)
    candidates: Final = tuple(candidate(row, ()) for row in rows.get("data", ()))
    chat: Final = tuple(c for c in candidates if c.mode == "chat")
    eligible: Final = tuple(c for c in chat if matches(c, request.requirements))
    group_names: Final = {
        c.model_group: (
            resolve_model_group_alias(llm_router.registry().aliases(), lookup[c.model_group]) or lookup[c.model_group]
        )
        if llm_router
        else lookup[c.model_group]
        for c in eligible
    }
    resolved: Final = tuple(
        candidate(
            c, llm_router.registry().deployments(group_names[c.model_group], team_id=auth.team_id) if llm_router else ()
        )
        for c in eligible
    )
    benchmarks: Final = await cache.get() if request.priorities.intelligence and request.focus else ()
    speeds: Final = await speed_cache.get(client, tuple(group_names.values())) if request.priorities.speed else ()
    speed_by_group: Final = {speed.model_group: speed for speed in speeds}
    now: Final = datetime.now(timezone.utc).date().isoformat()
    return RecommendationResponse(
        recommendations=rank(
            resolved,
            request,
            benchmarks,
            {
                public: speed_by_group[internal]
                for public, internal in group_names.items()
                if internal in speed_by_group
            },
        ),
        candidates_considered=len(chat),
        candidates_filtered=len(chat) - len(eligible),
        data_sources=(
            DataSource(
                name="Epoch AI Benchmarking Hub",
                url=EPOCH_URL,
                as_of=cache.as_of,
                available=bool(benchmarks),
                description="CC BY 4.0. Best exact-id score across effort variants.",
            ),
            DataSource(
                name="Gateway speed",
                as_of=now if speeds else None,
                available=bool(speeds),
                description="Last 7 days, at most the latest 100,000 gateway requests. Median latency and output tokens/sec.",
            ),
            DataSource(
                name="Gateway pricing",
                as_of=now,
                available=True,
                description="Configured model group prices, blended 3:1 input to output per million tokens.",
            ),
        ),
    )
