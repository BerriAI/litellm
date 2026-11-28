import pytest

from litellm.proxy._types import Litellm_EntityType
from litellm.proxy.hooks.model_max_budget_limiter import (
    model_budget_spend_cache_key,
    model_budget_start_time_cache_key,
)


@pytest.mark.parametrize(
    ("scope", "spend_prefix", "start_prefix"),
    [
        (Litellm_EntityType.KEY, "virtual_key_spend", "virtual_key_budget_start_time"),
        (Litellm_EntityType.USER, "user_model_spend", "user_model_budget_start_time"),
        (Litellm_EntityType.TEAM, "team_model_spend", "team_model_budget_start_time"),
        (Litellm_EntityType.END_USER, "end_user_model_spend", "end_user_budget_start_time"),
        (Litellm_EntityType.TEAM_MEMBER, "team_member_model_spend", "team_member_model_budget_start_time"),
    ],
)
def test_model_budget_windows_use_the_shared_redis_keys(
    scope: Litellm_EntityType, spend_prefix: str, start_prefix: str
) -> None:
    assert model_budget_spend_cache_key(scope, "caller", "claude*", "1h") == f"{spend_prefix}:caller:claude*:1h"
    assert model_budget_start_time_cache_key(scope, "caller", "claude*", "1h") == f"{start_prefix}:caller:claude*:1h"
