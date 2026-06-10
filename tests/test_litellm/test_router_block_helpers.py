"""Unit tests for Router block helper methods (coverage gate)."""

import pytest
import litellm
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


class TestRaiseModelBlockedError:
    def test_raises_permission_denied_error(self):
        router = _make_router("gpt-4o")
        with pytest.raises(litellm.PermissionDeniedError) as exc_info:
            router._raise_model_blocked_error("gpt-4o")
        assert exc_info.value.status_code == 403
        assert "Model is blocked" in str(exc_info.value)
