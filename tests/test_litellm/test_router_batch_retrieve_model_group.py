"""
model_group attribution on router batch retrieval.

Batch token usage is accounted on the *retrieve* call, not on create: the
provider only knows the token counts once the job finishes, so
`LiteLLMBatch.usage` arrives on `aretrieve_batch` and that is the record the
spend log tokens land on.

`aretrieve_batch` is addressed by batch_id, so the request carries no model,
and the router fans the lookup out across its deployments. These tests lock
that the winning deployment's model group is stamped on the emitted
StandardLoggingPayload, so `/global/activity/model` - which groups the spend
logs by `model_group` - can attribute those tokens instead of bucketing every
batch under "".
"""

import asyncio
from unittest.mock import MagicMock, patch

import pytest

import litellm
import litellm.batches.main as bm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import LiteLLMBatch, Usage

MODEL_GROUP = "vertex-gemini-2.5-flash-lite-dev"
DEPLOYMENT_MODEL = "vertex_ai/gemini-2.5-flash-lite"


class _PayloadCollector(CustomLogger):
    def __init__(self):
        super().__init__()
        self.payloads = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.payloads.append(kwargs.get("standard_logging_object"))


@pytest.fixture
def router():
    return Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": DEPLOYMENT_MODEL,
                    "vertex_project": "fake-project",
                    "vertex_location": "us-central1",
                    "vertex_credentials": "fake-creds",
                },
            }
        ]
    )


@pytest.fixture
def collector():
    logger = _PayloadCollector()
    previous = litellm.callbacks
    litellm.callbacks = [logger]
    try:
        yield logger
    finally:
        litellm.callbacks = previous


@pytest.fixture
def vertex_retrieve():
    """Mock the vertex provider seam - the only real network boundary."""
    batch = LiteLLMBatch(
        id="batch-1",
        completion_window="24h",
        created_at=0,
        endpoint="/v1/chat/completions",
        input_file_id="file-1",
        object="batch",
        status="completed",
        usage=Usage(prompt_tokens=1000, completion_tokens=200, total_tokens=1200),
    )
    seam = MagicMock(name="vertex_ai_batches_instance")
    seam.retrieve_batch.return_value = batch
    with patch.object(bm, "vertex_ai_batches_instance", seam):
        yield seam


async def _collected_payload(collector) -> dict:
    for _ in range(50):  # the success handler runs as a background task
        payloads = [p for p in collector.payloads if p is not None]
        if payloads:
            return payloads[-1]
        await asyncio.sleep(0.05)
    raise AssertionError(f"no StandardLoggingPayload was emitted: {collector.payloads}")


@pytest.mark.asyncio
async def test_aretrieve_batch_without_model_stamps_model_group(router, collector, vertex_retrieve):
    """
    The proxy retrieves a managed batch by id only - no `model` in the request.
    The router fans out over its deployments, so the model group is only known
    from the deployment that answered.
    """
    response = await router.aretrieve_batch(batch_id="batch-1")

    assert response.usage.total_tokens == 1200
    payload = await _collected_payload(collector)
    assert payload["model"] == DEPLOYMENT_MODEL
    assert payload["model_group"] == MODEL_GROUP


@pytest.mark.asyncio
async def test_aretrieve_batch_with_model_stamps_requested_model_group(router, collector, vertex_retrieve):
    """An explicitly requested model group is what gets logged."""
    await router.aretrieve_batch(model=MODEL_GROUP, batch_id="batch-1")

    payload = await _collected_payload(collector)
    assert payload["model_group"] == MODEL_GROUP
