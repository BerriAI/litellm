# What is this?
## Common auth checks between jwt + key based auth
"""
Got Valid Token from Cache, DB
Run checks for: 

1. If user can call model
2. If user is in budget 
3. If end_user ('user' passed to /chat/completions, /embeddings endpoint) is in budget 
"""
from litellm.proxy._types import LiteLLM_UserTable, LiteLLM_EndUserTable
from typing import Optional
from litellm.proxy.utils import PrismaClient
from litellm.caching import DualCache


def common_checks(
    request_body: dict,
    user_object: LiteLLM_UserTable,
    end_user_object: Optional[LiteLLM_EndUserTable],
) -> bool:
    _model = request_body.get("model", None)
    # 1. If user can call model
    if (
        _model is not None
        and len(user_object.models) > 0
        and _model not in user_object.models
    ):
        raise Exception(
            f"User={user_object.user_id} not allowed to call model={_model}. Allowed user models = {user_object.models}"
        )
    # 2. If user is in budget
    if (
        user_object.max_budget is not None
        and user_object.spend > user_object.max_budget
    ):
        raise Exception(
            f"User={user_object.user_id} over budget. Spend={user_object.spend}, Budget={user_object.max_budget}"
        )
    # 3. If end_user ('user' passed to /chat/completions, /embeddings endpoint) is in budget
    if end_user_object is not None and end_user_object.litellm_budget_table is not None:
        end_user_budget = end_user_object.litellm_budget_table.max_budget
        if end_user_budget is not None and end_user_object.spend > end_user_budget:
            raise Exception(
                f"End User={end_user_object.user_id} over budget. Spend={end_user_object.spend}, Budget={end_user_budget}"
            )
    return True


async def get_end_user_object(
    end_user_id: Optional[str],
    prisma_client: Optional[PrismaClient],
    user_api_key_cache: DualCache,
) -> Optional[LiteLLM_EndUserTable]:
    """
    Returns end user object, if in db.

    Do a isolated check for end user in table vs. doing a combined key + team + user + end-user check, as key might come in frequently for different end-users. Larger call will slowdown query time. This way we get to cache the constant (key/team/user info) and only update based on the changing value (end-user).
    """
    if prisma_client is None:
        raise Exception("No db connected")

    if end_user_id is None:
        return None

    # check if in cache
    cached_user_obj = user_api_key_cache.async_get_cache(key=end_user_id)
    if cached_user_obj is not None:
        if isinstance(cached_user_obj, dict):
            return LiteLLM_EndUserTable(**cached_user_obj)
        elif isinstance(cached_user_obj, LiteLLM_EndUserTable):
            return cached_user_obj
    # else, check db
    try:
        response = await prisma_client.db.litellm_endusertable.find_unique(
            where={"user_id": end_user_id}
        )

        if response is None:
            raise Exception

        return LiteLLM_EndUserTable(**response.dict())
    except Exception as e:  # if end-user not in db
        return None
