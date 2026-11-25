from types import SimpleNamespace
from typing import Final, Literal

import pytest
from fastapi import HTTPException

from litellm.litellm_core_utils.core_helpers import require_attributes
from litellm.models.budget import LiteLLM_BudgetTable
from litellm.proxy.auth.auth_checks import (
    get_default_end_user_budget,
    get_model_access_group_budgets_batch,
    get_team_member_default_budget,
)
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.utils import PrismaClient
from litellm.types.proxy.model_access_group_budget import ModelAccessGroupBudget
from tests.unit.proxy.rust_auth_stores import GroupBudget, RustAuthStores


@pytest.mark.asyncio
@pytest.mark.parametrize("reader", ("end_user", "team_member", "model_access_group"))
@pytest.mark.parametrize("has_windows", (False, True))
async def test_scalar_readers_reject_windows_instead_of_returning_no_budget(
    reader: Literal["end_user", "team_member", "model_access_group"],
    has_windows: bool,
    rust_auth_stores: RustAuthStores,
) -> None:
    rust_auth_stores.budget.rows["budget"] = {
        "budget_id": "budget",
        "max_budget": 0,
        "budget_limits": [{"budget_duration": "1mo", "max_budget": 0}] if has_windows else None,
    }
    rust_auth_stores.model_access_group.groups["group"] = GroupBudget(max_budget=0, has_budget_windows=has_windows)
    client: Final = require_attributes(SimpleNamespace(db=SimpleNamespace()), PrismaClient, "db")
    cache: Final = UserApiKeyCache()

    async def read() -> object:
        if reader == "model_access_group":
            return await get_model_access_group_budgets_batch(("group",), client, cache)
        loader: Final = get_default_end_user_budget if reader == "end_user" else get_team_member_default_budget
        return await loader(budget_id="budget", prisma_client=client, user_api_key_cache=cache)

    if has_windows:
        with pytest.raises(HTTPException, match="customer hierarchies") as error:
            await read()
        assert error.value.status_code == 400
    else:
        result: Final = await read()
        expected: Final = (
            {"group": ModelAccessGroupBudget(access_group_name="group", spend=0, max_budget=0)}
            if reader == "model_access_group"
            else LiteLLM_BudgetTable(budget_id="budget", max_budget=0)
        )
        assert result == expected
