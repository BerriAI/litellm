"""Unit tests for Router block helper methods (coverage gate)."""

from litellm import Router


def _make_router(model_name: str, blocked: bool = False) -> Router:
    return Router(
        model_list=[
            {
                "model_name": model_name,
                "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                "model_info": {"blocked": blocked},
            }
        ]
    )


class TestAreAllDeploymentsBlocked:
    def test_all_blocked_returns_true(self):
        router = _make_router("gpt-4o", blocked=True)
        deployments = router.get_model_list(model_name="gpt-4o") or []
        assert router._are_all_deployments_blocked(deployments) is True

    def test_one_not_blocked_returns_false(self):
        router = Router(
            model_list=[
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
                    "model_info": {"blocked": True},
                },
                {
                    "model_name": "gpt-4o",
                    "litellm_params": {
                        "model": "openai/gpt-4o-mini",
                        "api_key": "fake",
                    },
                    "model_info": {"blocked": False},
                },
            ]
        )
        deployments = router.get_model_list(model_name="gpt-4o") or []
        assert router._are_all_deployments_blocked(deployments) is False

    def test_empty_list_returns_false(self):
        router = _make_router("gpt-4o")
        assert router._are_all_deployments_blocked([]) is False


class TestIsModelFullyBlocked:
    def test_all_deployments_blocked_returns_true(self):
        router = _make_router("gpt-4o", blocked=True)
        assert router._is_model_fully_blocked("gpt-4o") is True

    def test_unblocked_deployment_returns_false(self):
        router = _make_router("gpt-4o", blocked=False)
        assert router._is_model_fully_blocked("gpt-4o") is False


def _deployment(model_name: str, dep_id: str, blocked: bool) -> dict:
    return {
        "model_name": model_name,
        "litellm_params": {"model": "openai/gpt-4o", "api_key": "fake"},
        "model_info": {"id": dep_id, "blocked": blocked},
    }


class TestModelGroupHasUnblockedDeployment:
    def test_one_unblocked_returns_true(self):
        router = Router(
            model_list=[
                _deployment("group", "d0", blocked=True),
                _deployment("group", "d1", blocked=False),
            ]
        )
        assert router._model_group_has_unblocked_deployment("group", team_id=None) is True

    def test_all_blocked_returns_false(self):
        router = Router(model_list=[_deployment("group", "d0", blocked=True)])
        assert router._model_group_has_unblocked_deployment("group", team_id=None) is False

    def test_unknown_group_returns_false(self):
        router = _make_router("group", blocked=False)
        assert router._model_group_has_unblocked_deployment("missing", team_id=None) is False


class TestHasReachableFallback:
    def test_reachable_when_direct_fallback_has_unblocked_deployment(self):
        router = Router(
            model_list=[
                _deployment("primary", "p0", blocked=True),
                _deployment("fallback", "f0", blocked=False),
            ]
        )
        assert router._has_reachable_fallback("primary", fallbacks=[{"primary": ["fallback"]}]) is True

    def test_not_reachable_when_fallback_also_fully_blocked(self):
        router = Router(
            model_list=[
                _deployment("primary", "p0", blocked=True),
                _deployment("fallback", "f0", blocked=True),
            ]
        )
        assert router._has_reachable_fallback("primary", fallbacks=[{"primary": ["fallback"]}]) is False

    def test_not_reachable_without_fallbacks(self):
        router = Router(model_list=[_deployment("primary", "p0", blocked=True)])
        assert router._has_reachable_fallback("primary", fallbacks=[]) is False

    def test_reachable_through_multi_level_chain(self):
        router = Router(
            model_list=[
                _deployment("primary", "p0", blocked=True),
                _deployment("mid", "m0", blocked=True),
                _deployment("healthy", "h0", blocked=False),
            ]
        )
        chain = [{"primary": ["mid"]}, {"mid": ["healthy"]}]
        assert router._has_reachable_fallback("primary", fallbacks=chain) is True

    def test_self_referential_chain_terminates(self):
        router = Router(
            model_list=[
                _deployment("primary", "p0", blocked=True),
                _deployment("fallback", "f0", blocked=True),
            ]
        )
        chain = [{"primary": ["fallback"]}, {"fallback": ["primary"]}]
        assert router._has_reachable_fallback("primary", fallbacks=chain) is False

    def test_generic_star_fallback_is_honored(self):
        router = Router(
            model_list=[
                _deployment("primary", "p0", blocked=True),
                _deployment("fallback", "f0", blocked=False),
            ]
        )
        assert router._has_reachable_fallback("primary", fallbacks=[{"*": ["fallback"]}]) is True

    def test_string_form_fallback_is_honored(self):
        router = Router(
            model_list=[
                _deployment("primary", "p0", blocked=True),
                _deployment("fallback", "f0", blocked=False),
            ]
        )
        assert router._has_reachable_fallback("primary", fallbacks=["fallback"]) is True

    def test_chain_longer_than_max_fallbacks_fails_closed(self):
        hops = 6
        router = Router(
            model_list=[_deployment("m0", "m0", blocked=True)]
            + [_deployment(f"m{i}", f"m{i}", blocked=True) for i in range(1, hops)]
            + [_deployment("healthy", "h0", blocked=False)],
            max_fallbacks=hops - 2,
        )
        chain = [{f"m{i}": [f"m{i + 1}"]} for i in range(hops - 1)] + [{f"m{hops - 1}": ["healthy"]}]
        assert router._has_reachable_fallback("m0", fallbacks=chain) is False


class TestIsBlockedWithoutReachableFallback:
    def test_blocked_and_no_fallback_returns_true(self):
        router = Router(model_list=[_deployment("primary", "p0", blocked=True)])
        assert router._is_blocked_without_reachable_fallback("primary", reachable_fallbacks=None, team_id=None) is True

    def test_not_all_blocked_returns_false(self):
        router = Router(model_list=[_deployment("primary", "p0", blocked=False)])
        assert router._is_blocked_without_reachable_fallback("primary", reachable_fallbacks=None, team_id=None) is False

    def test_blocked_with_reachable_fallback_returns_false(self):
        router = Router(
            model_list=[
                _deployment("primary", "p0", blocked=True),
                _deployment("fallback", "f0", blocked=False),
            ]
        )
        reachable = [{"primary": ["fallback"]}]
        assert (
            router._is_blocked_without_reachable_fallback("primary", reachable_fallbacks=reachable, team_id=None)
            is False
        )

    def test_blocked_with_fully_blocked_fallback_returns_true(self):
        router = Router(
            model_list=[
                _deployment("primary", "p0", blocked=True),
                _deployment("fallback", "f0", blocked=True),
            ]
        )
        reachable = [{"primary": ["fallback"]}]
        assert (
            router._is_blocked_without_reachable_fallback("primary", reachable_fallbacks=reachable, team_id=None)
            is True
        )
