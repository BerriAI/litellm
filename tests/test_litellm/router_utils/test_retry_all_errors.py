"""Tests that should_retry_this_error never exits early when deployments exist."""
import pytest
from unittest.mock import MagicMock

import httpx
import litellm
import openai


@pytest.fixture
def router():
    return litellm.Router(
        model_list=[
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "key1"},
                "model_info": {"id": "dep-1"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "key2"},
                "model_info": {"id": "dep-2"},
            },
            {
                "model_name": "gpt-4",
                "litellm_params": {"model": "gpt-4", "api_key": "key3"},
                "model_info": {"id": "dep-3"},
            },
        ],
    )


class TestRetryAllErrors:
    @pytest.mark.parametrize(
        "exception_class,status_code",
        [
            (litellm.BadRequestError, 400),
            (litellm.AuthenticationError, 401),
            (litellm.NotFoundError, 404),
            (litellm.UnprocessableEntityError, 422),
            (litellm.RateLimitError, 429),
            (litellm.InternalServerError, 500),
        ],
        ids=lambda x: str(x) if isinstance(x, int) else x.__name__,
    )
    def test_retries_all_error_types_when_healthy_deployments_exist(
        self, router, exception_class, status_code
    ):
        """Should NOT raise when healthy deployments exist, regardless of error type."""
        kwargs = dict(
            message="test error",
            model="gpt-4",
            llm_provider="openai",
        )
        if exception_class is litellm.UnprocessableEntityError:
            kwargs["response"] = httpx.Response(
                status_code=status_code,
                request=httpx.Request("POST", "https://api.openai.com"),
            )
        error = exception_class(**kwargs)
        # Should return True (allow retry), not raise
        result = router.should_retry_this_error(
            error=error,
            healthy_deployments=[{"id": "dep-2"}, {"id": "dep-3"}],
            all_deployments=[{"id": "dep-1"}, {"id": "dep-2"}, {"id": "dep-3"}],
        )
        assert result is True

    def test_raises_when_no_healthy_deployments(self, router):
        """Should raise when no healthy deployments remain."""
        error = litellm.InternalServerError(
            message="test error",
            model="gpt-4",
            llm_provider="openai",
        )
        with pytest.raises(Exception):
            router.should_retry_this_error(
                error=error,
                healthy_deployments=[],
                all_deployments=[{"id": "dep-1"}],
            )
