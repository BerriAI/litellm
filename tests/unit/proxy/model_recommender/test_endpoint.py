from collections.abc import Mapping, Sequence
from typing import Final

import pytest

from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.model_recommender.endpoint import recommend_models
from litellm.proxy.model_recommender.models import Priorities, RecommendationRequest
from litellm.proxy.model_recommender.speed import SpeedCache
from litellm.router import Router


class Database:
    async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
        groups: Final = args[0]
        return tuple(
            {"model_group": name, "latency_seconds": latency}
            for name, latency in (("chat", 2.0), ("model_name_own_chat", 3.0), ("model_name_other_chat", 99.0))
            if isinstance(groups, list) and name in groups
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(("internal", "latency"), (("chat", 2.0), ("model_name_own_chat", 3.0)))
async def test_telemetry_uses_the_listings_caller_scoped_lookup(internal: str, latency: float) -> None:
    router: Final = Router(
        model_list=[
            {"model_name": "chat", "litellm_params": {"model": "openai/global", "api_key": "test"}},
            {
                "model_name": "model_name_own_chat",
                "litellm_params": {"model": "openai/own", "api_key": "test"},
                "model_info": {"team_id": "own", "team_public_model_name": "chat"},
            },
            {
                "model_name": "model_name_other_chat",
                "litellm_params": {"model": "openai/other", "api_key": "test"},
                "model_info": {"team_id": "other", "team_public_model_name": "chat"},
            },
        ]
    )
    response: Final = await recommend_models(
        RecommendationRequest(priorities=Priorities(intelligence=0, speed=3, cost=0)),
        UserAPIKeyAuth(team_id="own"),
        {"data": ({"model_group": "chat", "mode": "chat"},)},
        router,
        {"chat": internal},
        Database(),
        speed_cache=SpeedCache(),
    )
    assert response.recommendations[0].model_group == "chat"
    assert response.recommendations[0].latency_seconds == latency


@pytest.mark.asyncio
async def test_zero_speed_priority_does_not_query_database() -> None:
    class UnexpectedDatabase:
        async def query_raw(self, query: str, *args: object) -> Sequence[Mapping[str, object]]:
            pytest.fail("speed-disabled recommendations queried telemetry")

    response: Final = await recommend_models(
        RecommendationRequest(priorities=Priorities(intelligence=0, speed=0, cost=3)),
        UserAPIKeyAuth(),
        {"data": ({"model_group": "chat", "mode": "chat"},)},
        None,
        {"chat": "chat"},
        UnexpectedDatabase(),
    )
    assert response.recommendations[0].signals.speed.unknown
    assert response.recommendations[0].latency_seconds is None
