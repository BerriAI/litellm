"""
model_group attribution on router batch retrieval.

Batch token usage is accounted on the *retrieve* call, not on create: a provider
only reports token counts once the job finishes, so the usage is read off the
completed batch's output file during retrieve logging and that is the spend log
row the tokens land on.

A batch is retrieved by id, so the request carries no model and the router fans
the lookup out across its deployments. These tests lock that the answering
deployment's model group is stamped on the emitted StandardLoggingPayload, so
`/global/activity/model` - which groups the spend logs by `model_group` - can
attribute those tokens instead of bucketing every batch under "".

The provider is faked at the HTTP boundary, so the whole retrieve + usage
accounting path runs for real.
"""

import asyncio
import json

import httpx
import pytest
import respx

import litellm
from litellm import Router
from litellm.integrations.custom_logger import CustomLogger

MODEL_GROUP = "gemini-batch-group"
DEPLOYMENT_MODEL = "openai/gpt-4o-mini"
API_BASE = "http://localhost:4001/v1"
BATCH_ID = "batch-1"
ROWS = 2
TOKENS_PER_ROW = 600

COMPLETED_BATCH = {
    "id": BATCH_ID,
    "object": "batch",
    "endpoint": "/v1/chat/completions",
    "errors": None,
    "input_file_id": "file-in-1",
    "completion_window": "24h",
    "status": "completed",
    "output_file_id": "file-out-1",
    "error_file_id": None,
    "created_at": 0,
    "completed_at": 1,
    "request_counts": {"total": ROWS, "completed": ROWS, "failed": 0},
    "metadata": None,
}

OUTPUT_JSONL = "\n".join(
    json.dumps(
        {
            "id": f"req-{row}",
            "custom_id": f"row-{row}",
            "response": {
                "status_code": 200,
                "body": {
                    "id": f"chatcmpl-{row}",
                    "object": "chat.completion",
                    "model": "gpt-4o-mini",
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}
                    ],
                    "usage": {"prompt_tokens": 500, "completion_tokens": 100, "total_tokens": TOKENS_PER_ROW},
                },
            },
        }
    )
    for row in range(ROWS)
)


class _PayloadCollector(CustomLogger):
    """Captures the StandardLoggingPayload the spend log is built from."""

    def __init__(self):
        super().__init__()
        self.payloads = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.payloads.append(kwargs.get("standard_logging_object"))

    async def retrieve_batch_payload(self) -> dict:
        for _ in range(100):  # the success handler runs as a background task
            for payload in self.payloads:
                if payload and payload.get("call_type") == "aretrieve_batch":
                    return payload
            await asyncio.sleep(0.05)
        raise AssertionError(f"no aretrieve_batch payload was emitted: {self.payloads}")


@pytest.fixture
def router():
    return Router(
        model_list=[
            {
                "model_name": MODEL_GROUP,
                "litellm_params": {
                    "model": DEPLOYMENT_MODEL,
                    "api_base": API_BASE,
                    "api_key": "sk-fake",
                },
            }
        ]
    )


@pytest.fixture
def collector(monkeypatch):
    logger = _PayloadCollector()
    monkeypatch.setattr(litellm, "callbacks", [logger])
    return logger


@pytest.fixture
def provider():
    """Fake the provider at the HTTP boundary: the completed batch plus the
    output file the usage accounting reads."""
    with respx.mock(assert_all_called=True) as respx_mock:
        respx_mock.get(f"{API_BASE}/batches/{BATCH_ID}").mock(return_value=httpx.Response(200, json=COMPLETED_BATCH))
        respx_mock.get(f"{API_BASE}/files/file-out-1/content").mock(return_value=httpx.Response(200, text=OUTPUT_JSONL))
        yield respx_mock


@pytest.mark.asyncio
async def test_aretrieve_batch_without_model_stamps_model_group(router, collector, provider):
    """
    The proxy retrieves a managed batch by id only - no `model` in the request.
    The router fans out over its deployments, so the model group is only known
    from the deployment that answered.
    """
    response = await router.aretrieve_batch(batch_id=BATCH_ID)

    assert response.id == BATCH_ID
    payload = await collector.retrieve_batch_payload()
    assert payload["total_tokens"] == ROWS * TOKENS_PER_ROW
    assert payload["model"] == DEPLOYMENT_MODEL
    assert payload["model_group"] == MODEL_GROUP


@pytest.mark.asyncio
async def test_aretrieve_batch_with_model_stamps_requested_model_group(router, collector, provider):
    """An explicitly requested model group is what gets logged."""
    await router.aretrieve_batch(model=MODEL_GROUP, batch_id=BATCH_ID)

    payload = await collector.retrieve_batch_payload()
    assert payload["model_group"] == MODEL_GROUP
