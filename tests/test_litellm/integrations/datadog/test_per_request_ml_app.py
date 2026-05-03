"""
Tests for per-request ml_app override in Datadog LLM Observability.

Verifies that callers can pass metadata.ml_app to control the Application
and Service columns in Datadog, allowing multiple services sharing a single
LiteLLM proxy to appear as distinct applications.
"""

import asyncio
import json
import os
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from litellm.integrations.datadog.datadog_llm_obs import DataDogLLMObsLogger
from litellm.types.utils import (
    StandardLoggingHiddenParams,
    StandardLoggingMetadata,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
)


def create_standard_logging_payload(**overrides) -> StandardLoggingPayload:
    """Create a minimal StandardLoggingPayload for testing."""
    base = StandardLoggingPayload(
        id="test-id",
        call_type="completion",
        response_cost=0.01,
        response_cost_failure_debug_info=None,
        status="success",
        total_tokens=100,
        prompt_tokens=50,
        completion_tokens=50,
        startTime=1000.0,
        endTime=1001.0,
        completionStartTime=1000.5,
        model_map_information=StandardLoggingModelInformation(
            model_map_key="gpt-4", model_map_value=None
        ),
        model="gpt-4",
        model_id="model-1",
        model_group="openai",
        api_base="https://api.openai.com",
        metadata=StandardLoggingMetadata(
            user_api_key_hash="hash",
            user_api_key_org_id=None,
            user_api_key_alias="alias",
            user_api_key_team_id=None,
            user_api_key_user_id=None,
            user_api_key_team_alias=None,
            spend_logs_metadata=None,
            requester_ip_address="127.0.0.1",
            requester_metadata=None,
        ),
        cache_hit=False,
        cache_key=None,
        saved_cache_cost=0.0,
        request_tags=[],
        end_user=None,
        requester_ip_address="127.0.0.1",
        messages=[{"role": "user", "content": "Hello"}],
        response={"choices": [{"message": {"role": "assistant", "content": "Hi"}}]},
        error_str=None,
        model_parameters={},
        hidden_params=StandardLoggingHiddenParams(
            model_id="model-1",
            cache_key=None,
            api_base="https://api.openai.com",
            response_cost="0.01",
            additional_headers=None,
        ),
        trace_id="trace-1",
        custom_llm_provider="openai",
    )
    base.update(overrides)
    return base


@pytest.fixture
def mock_env_vars():
    with patch.dict(
        os.environ, {"DD_API_KEY": "test-key", "DD_SITE": "us5.datadoghq.com"}
    ):
        yield


@pytest.fixture
def logger(mock_env_vars):
    with patch(
        "litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client"
    ), patch("asyncio.create_task"):
        return DataDogLLMObsLogger()


class TestPerRequestMlApp:
    """Test that metadata.ml_app overrides the Application and Service columns."""

    def test_payload_gets_dd_ml_app_from_metadata(self, logger):
        """When metadata contains ml_app, the payload should store it as _dd_ml_app."""
        kwargs = {
            "standard_logging_object": create_standard_logging_payload(),
            "litellm_params": {"metadata": {"ml_app": "dada"}},
        }
        payload = logger.create_llm_obs_payload(
            kwargs, datetime.now(), datetime.now() + timedelta(seconds=1)
        )

        assert payload.get("_dd_ml_app") == "dada"

    def test_payload_without_ml_app_has_no_dd_ml_app(self, logger):
        """When metadata does not contain ml_app, _dd_ml_app should not be set."""
        kwargs = {
            "standard_logging_object": create_standard_logging_payload(),
            "litellm_params": {"metadata": {}},
        }
        payload = logger.create_llm_obs_payload(
            kwargs, datetime.now(), datetime.now() + timedelta(seconds=1)
        )

        assert "_dd_ml_app" not in payload

    @pytest.mark.asyncio
    async def test_dd_llmobs_ml_app_env_var_used_as_default(self, mock_env_vars):
        """DD_LLMOBS_ML_APP env var should be used as the default ml_app."""
        with patch.dict(os.environ, {"DD_LLMOBS_ML_APP": "my-app"}), patch(
            "litellm.integrations.datadog.datadog_llm_obs.get_async_httpx_client"
        ), patch("asyncio.create_task"):
            logger = DataDogLLMObsLogger()

        now = datetime.now()
        # Span without per-request ml_app — should use DD_LLMOBS_ML_APP
        kwargs = {
            "standard_logging_object": create_standard_logging_payload(),
            "litellm_params": {"metadata": {}},
        }
        payload = logger.create_llm_obs_payload(
            kwargs, now, now + timedelta(seconds=1)
        )
        logger.log_queue.append(payload)

        sent_payloads = []

        async def capture_post(**kwargs):
            resp = MagicMock()
            resp.status_code = 202
            sent_payloads.append(json.loads(kwargs.get("content", "")))
            return resp

        logger.async_client = MagicMock()
        logger.async_client.post = AsyncMock(side_effect=capture_post)

        with patch.dict(os.environ, {"DD_LLMOBS_ML_APP": "my-app"}):
            await logger.async_send_batch()

        assert len(sent_payloads) == 1
        assert sent_payloads[0]["data"]["attributes"]["ml_app"] == "my-app"

    @pytest.mark.asyncio
    async def test_batch_grouped_by_ml_app(self, logger):
        """Spans with different ml_app values should be sent as separate batches."""
        now = datetime.now()

        # Create two spans with different ml_app values
        for ml_app in ["dada", "navi"]:
            kwargs = {
                "standard_logging_object": create_standard_logging_payload(
                    id=f"req-{ml_app}", trace_id=f"trace-{ml_app}"
                ),
                "litellm_params": {"metadata": {"ml_app": ml_app}},
            }
            payload = logger.create_llm_obs_payload(
                kwargs, now, now + timedelta(seconds=1)
            )
            logger.log_queue.append(payload)

        # Add one span without ml_app (should use default)
        kwargs = {
            "standard_logging_object": create_standard_logging_payload(
                id="req-default", trace_id="trace-default"
            ),
            "litellm_params": {"metadata": {}},
        }
        payload = logger.create_llm_obs_payload(
            kwargs, now, now + timedelta(seconds=1)
        )
        logger.log_queue.append(payload)

        # Track all POST calls
        sent_payloads = []

        async def capture_post(**kwargs):
            resp = MagicMock()
            resp.status_code = 202
            content = kwargs.get("content", "")
            sent_payloads.append(json.loads(content))
            return resp

        logger.async_client = MagicMock()
        logger.async_client.post = AsyncMock(side_effect=capture_post)

        await logger.async_send_batch()

        # Should have sent 3 batches: dada, navi, and default
        assert len(sent_payloads) == 3

        ml_apps = {p["data"]["attributes"]["ml_app"] for p in sent_payloads}
        assert "dada" in ml_apps
        assert "navi" in ml_apps

        # Verify _dd_ml_app is stripped from the spans
        for p in sent_payloads:
            for span in p["data"]["attributes"]["spans"]:
                assert "_dd_ml_app" not in span

    @pytest.mark.asyncio
    async def test_batch_single_ml_app_sends_one_request(self, logger):
        """When all spans have the same ml_app, only one batch is sent."""
        now = datetime.now()

        for i in range(3):
            kwargs = {
                "standard_logging_object": create_standard_logging_payload(
                    id=f"req-{i}", trace_id=f"trace-{i}"
                ),
                "litellm_params": {"metadata": {"ml_app": "dada"}},
            }
            payload = logger.create_llm_obs_payload(
                kwargs, now, now + timedelta(seconds=1)
            )
            logger.log_queue.append(payload)

        sent_payloads = []

        async def capture_post(**kwargs):
            resp = MagicMock()
            resp.status_code = 202
            sent_payloads.append(json.loads(kwargs.get("content", "")))
            return resp

        logger.async_client = MagicMock()
        logger.async_client.post = AsyncMock(side_effect=capture_post)

        await logger.async_send_batch()

        assert len(sent_payloads) == 1
        assert sent_payloads[0]["data"]["attributes"]["ml_app"] == "dada"
        assert len(sent_payloads[0]["data"]["attributes"]["spans"]) == 3

    @pytest.mark.asyncio
    async def test_batch_non_202_response_does_not_clear_queue(self, logger):
        """A non-202 response should log the error and keep spans in the queue."""
        now = datetime.now()

        kwargs = {
            "standard_logging_object": create_standard_logging_payload(),
            "litellm_params": {"metadata": {"ml_app": "dada"}},
        }
        payload = logger.create_llm_obs_payload(
            kwargs, now, now + timedelta(seconds=1)
        )
        logger.log_queue.append(payload)

        async def bad_post(**kwargs):
            resp = MagicMock()
            resp.status_code = 400
            resp.text = "Bad Request"
            return resp

        logger.async_client = MagicMock()
        logger.async_client.post = AsyncMock(side_effect=bad_post)

        # Should not raise — error is caught and logged
        await logger.async_send_batch()

        # Queue should NOT be cleared on failure
        assert len(logger.log_queue) == 1

    @pytest.mark.asyncio
    async def test_queue_cleared_after_batch(self, logger):
        """The log queue should be empty after a successful batch send."""
        now = datetime.now()

        kwargs = {
            "standard_logging_object": create_standard_logging_payload(),
            "litellm_params": {"metadata": {"ml_app": "dada"}},
        }
        payload = logger.create_llm_obs_payload(
            kwargs, now, now + timedelta(seconds=1)
        )
        logger.log_queue.append(payload)

        async def ok_post(**kwargs):
            resp = MagicMock()
            resp.status_code = 202
            return resp

        logger.async_client = MagicMock()
        logger.async_client.post = AsyncMock(side_effect=ok_post)

        await logger.async_send_batch()

        assert len(logger.log_queue) == 0
