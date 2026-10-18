import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from litellm.types.router import GenericLiteLLMParams


@pytest.mark.asyncio
async def test_team_admin_cannot_move_api_base_keeping_server_key():
    from fastapi import HTTPException
    from litellm.proxy._types import LitellmUserRoles, ProxyException, UserAPIKeyAuth
    from litellm.types.router import GenericLiteLLMParams

    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelManagementAuthChecks,
    )

    stored = GenericLiteLLMParams(api_key="server-owned-key")
    incoming = GenericLiteLLMParams(api_base="https://attacker.example")
    model_params = MagicMock()
    model_params.litellm_params = stored

    with pytest.raises(ProxyException) as exc:
        await ModelManagementAuthChecks._reject_non_admin_wif_write(
            model_params=model_params,
            incoming_params=incoming.model_dump(exclude_none=True),
            user_api_key_dict=UserAPIKeyAuth(
                api_key="sk-team-admin",
                user_role=LitellmUserRoles.INTERNAL_USER,
                user_id="team-admin-1",
            ),
            prisma_client=AsyncMock(),
        )
    assert int(getattr(exc.value, "code", 0)) == 403
    assert "api_key stays server-owned" in str(getattr(exc.value, "message", getattr(exc.value, "detail", "")))


@pytest.mark.asyncio
async def test_team_admin_bring_your_own_key_move_allowed():
    import litellm.proxy.management_endpoints.model_management_endpoints as mme
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelManagementAuthChecks,
    )
    from litellm.types.router import GenericLiteLLMParams
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    stored = GenericLiteLLMParams(api_key="server-owned-key")
    incoming = GenericLiteLLMParams(api_base="https://my-team-proxy.example", api_key="my-team-key")
    model_params = MagicMock()
    model_params.litellm_params = stored

    with (
        patch.object(mme, "effective_server_owned_wif_fields", AsyncMock(return_value=())),
    ):
        await ModelManagementAuthChecks._reject_non_admin_wif_write(
            model_params=model_params,
            incoming_params=incoming.model_dump(exclude_none=True),
            user_api_key_dict=UserAPIKeyAuth(
                api_key="sk-team",
                user_role=LitellmUserRoles.INTERNAL_USER,
                user_id="team-admin-1",
            ),
            prisma_client=AsyncMock(),
        )


@pytest.mark.asyncio
async def test_team_admin_wif_association_still_guarded():
    """Under WIF the original server-owned-token rule keeps firing."""
    import litellm.proxy.management_endpoints.model_management_endpoints as mme
    from litellm.proxy.management_endpoints.model_management_endpoints import (
        ModelManagementAuthChecks,
    )
    from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth

    stored = GenericLiteLLMParams(api_key="server-key")
    incoming = GenericLiteLLMParams(api_base="https://moved.example")
    model_params = MagicMock()
    model_params.litellm_params = stored

    with (
        patch.object(mme, "effective_server_owned_wif_fields", AsyncMock(return_value=("oidc_token_url",))),
    ):
        # Capture outside the except block: on this stack both the static-key
        # harvest gate and the WIF rule can fire for the same shape; accept
        # either interdiction.
        guard_raised = None
        try:
            await ModelManagementAuthChecks._reject_non_admin_wif_write(
                model_params=model_params,
                incoming_params=incoming.model_dump(exclude_none=True),
                user_api_key_dict=UserAPIKeyAuth(
                    api_key="sk-team", user_role=LitellmUserRoles.INTERNAL_USER, user_id="t"
                ),
                prisma_client=AsyncMock(),
            )
        except Exception as guard_error:
            guard_raised = guard_error
        assert guard_raised is not None, "expected a server-owned-key interdiction to fire"
        assert ("api_key stays server-owned" in str(guard_raised)) or ("workload identity" in str(guard_raised))
