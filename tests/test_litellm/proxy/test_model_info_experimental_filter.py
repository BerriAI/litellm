"""
Tests for the `experimental` tag filter on the /model/info endpoint (model_info_v1).

Deployments tagged "experimental" / "experimental:<id>" in litellm_params.tags are
hidden from the listing by default for every caller. Only a PROXY_ADMIN that
explicitly passes include_experimental=true gets them back.
"""

from typing import List, Optional
from unittest.mock import MagicMock, patch

import pytest

from litellm.proxy._types import LitellmUserRoles, UserAPIKeyAuth
from litellm.proxy.proxy_server import _is_experimental_deployment
from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo


def _deployment_dict(model_name: str, tags: Optional[List[str]] = None) -> dict:
    params: dict = {"model": f"openai/{model_name}"}
    if tags is not None:
        params["tags"] = tags
    return Deployment(
        model_name=model_name,
        litellm_params=LiteLLM_Params(**params),
        model_info=ModelInfo(),
    ).model_dump(exclude_none=True)


class TestIsExperimentalDeployment:
    def test_plain_experimental_tag(self):
        assert _is_experimental_deployment(_deployment_dict("m", ["experimental"])) is True

    def test_experimental_with_id(self):
        assert _is_experimental_deployment(_deployment_dict("m", ["prod", "experimental:501"])) is True

    def test_no_tags(self):
        assert _is_experimental_deployment(_deployment_dict("m")) is False

    def test_unrelated_tags(self):
        assert _is_experimental_deployment(_deployment_dict("m", ["prod", "beta"])) is False

    def test_lookalike_tag_not_matched(self):
        # "experimental_foo" is not an experimental marker (needs exact or "experimental:")
        assert _is_experimental_deployment(_deployment_dict("m", ["experimental_foo"])) is False


async def _call_model_info(user_api_key_dict, include_experimental):
    from litellm.proxy.proxy_server import model_info_v1

    normal = _deployment_dict("normal-model")
    exp = _deployment_dict("exp-model", ["experimental:501"])
    by_name = {"normal-model": [normal], "exp-model": [exp]}

    mock_router = MagicMock()
    mock_router.model_list = [normal, exp]
    mock_router.get_model_names.return_value = ["normal-model", "exp-model"]
    mock_router.get_model_access_groups.return_value = {}
    mock_router.get_model_list_from_model_alias.return_value = []
    mock_router.get_model_list.side_effect = lambda model_name: by_name.get(model_name)

    with patch("litellm.proxy.proxy_server.llm_router", mock_router), \
         patch("litellm.proxy.proxy_server.llm_model_list", [normal, exp]), \
         patch("litellm.proxy.proxy_server.prisma_client", None), \
         patch("litellm.proxy.proxy_server.user_model", None), \
         patch("litellm.proxy.proxy_server.get_key_models", return_value=["normal-model", "exp-model"]), \
         patch("litellm.proxy.proxy_server.get_team_models", return_value=["normal-model", "exp-model"]), \
         patch(
             "litellm.proxy.proxy_server.get_complete_model_list",
             return_value=["normal-model", "exp-model"],
         ):
        response = await model_info_v1(
            user_api_key_dict=user_api_key_dict,
            litellm_model_id=None,
            include_experimental=include_experimental,
        )
    return [m["model_name"] for m in response["data"]]


def _admin():
    return UserAPIKeyAuth(api_key="sk-test", user_role=LitellmUserRoles.PROXY_ADMIN)


def _non_admin():
    return UserAPIKeyAuth(api_key="sk-test")


class TestFullyExperimentalModelNames:
    """
    _fully_experimental_model_names powers the /v1/models and /models filter: a
    name is hidden only when EVERY deployment for it is experimental.
    """

    def _router(self, deployments):
        r = MagicMock()
        r.model_list = deployments
        return r

    def test_all_deployments_experimental_hidden(self):
        from litellm.proxy.proxy_server import _fully_experimental_model_names

        r = self._router([
            _deployment_dict("m1", ["experimental:1"]),
            _deployment_dict("m1", ["experimental:2"]),
        ])
        assert _fully_experimental_model_names(r) == {"m1"}

    def test_partial_replica_kept(self):
        """One replica down, one live -> the name stays visible."""
        from litellm.proxy.proxy_server import _fully_experimental_model_names

        r = self._router([
            _deployment_dict("m1", ["experimental:1"]),
            _deployment_dict("m1", None),
        ])
        assert _fully_experimental_model_names(r) == set()

    def test_no_experimental(self):
        from litellm.proxy.proxy_server import _fully_experimental_model_names

        r = self._router([_deployment_dict("m1"), _deployment_dict("m2", ["prod"])])
        assert _fully_experimental_model_names(r) == set()

    def test_mixed_names(self):
        from litellm.proxy.proxy_server import _fully_experimental_model_names

        r = self._router([
            _deployment_dict("down", ["experimental:9"]),
            _deployment_dict("up", None),
            _deployment_dict("up", ["experimental:9"]),  # one replica down, still up
        ])
        assert _fully_experimental_model_names(r) == {"down"}

    def test_none_router(self):
        from litellm.proxy.proxy_server import _fully_experimental_model_names

        assert _fully_experimental_model_names(None) == set()


class TestModelInfoExperimentalFilter:
    @pytest.mark.asyncio
    async def test_hidden_by_default_for_admin(self):
        names = await _call_model_info(_admin(), include_experimental=False)
        assert "normal-model" in names
        assert "exp-model" not in names

    @pytest.mark.asyncio
    async def test_admin_can_include(self):
        names = await _call_model_info(_admin(), include_experimental=True)
        assert "normal-model" in names
        assert "exp-model" in names

    @pytest.mark.asyncio
    async def test_hidden_by_default_for_non_admin(self):
        names = await _call_model_info(_non_admin(), include_experimental=False)
        assert "exp-model" not in names

    @pytest.mark.asyncio
    async def test_non_admin_cannot_include(self):
        # a non-admin passing include_experimental=true still gets them hidden
        names = await _call_model_info(_non_admin(), include_experimental=True)
        assert "normal-model" in names
        assert "exp-model" not in names
