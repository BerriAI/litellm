"""
Tests for litellm/batches/batch_line_item_logging.py and its hook in
Logging._async_success_handler_body.

When ``litellm.store_batch_line_items_in_callbacks`` is on and a completed
batch is logged (call_type aretrieve_batch), litellm emits one callback event
per JSONL line (request paired with response/error) in addition to the
aggregate batch event. These tests run the real Logging.async_success_handler
the way the CheckBatchCost poller invokes it, with a recording CustomLogger on
the async success/failure lists, so a regression in pairing, hidden params,
cost, or error propagation fails here.
"""

import json
import uuid
from datetime import datetime
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import LiteLLMBatch, Usage

INPUT_JSONL = b"\n".join(
    [
        json.dumps(
            {
                "custom_id": "a",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi a"}],
                    "temperature": 0.2,
                },
            }
        ).encode(),
        json.dumps(
            {
                "custom_id": "b",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "gpt-4o",
                    "messages": [{"role": "user", "content": "hi b"}],
                },
            }
        ).encode(),
    ]
)

OUTPUT_JSONL = json.dumps(
    {
        "custom_id": "a",
        "response": {
            "status_code": 200,
            "body": {
                "id": "chatcmpl-1",
                "model": "gpt-4o",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hello back"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        },
    }
).encode()

ERROR_JSONL = json.dumps(
    {
        "custom_id": "b",
        "response": {"status_code": 400, "body": {"error": {"message": "bad request boom"}}},
        "error": {"message": "bad request boom"},
    }
).encode()

_FILE_BYTES = {
    "input-file-1": INPUT_JSONL,
    "output-file-1": OUTPUT_JSONL,
    "error-file-1": ERROR_JSONL,
}


def _batch() -> LiteLLMBatch:
    return LiteLLMBatch(
        id="batch_1",
        object="batch",
        endpoint="/v1/chat/completions",
        input_file_id="input-file-1",
        output_file_id="output-file-1",
        error_file_id="error-file-1",
        status="completed",
        completion_window="24h",
        created_at=1,
    )


def _file_content(file_id: str, **_kwargs):
    return SimpleNamespace(content=_FILE_BYTES[file_id])


class _RecordingLogger(CustomLogger):
    def __init__(self) -> None:
        super().__init__()
        self.success_events = []
        self.failure_events = []

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self.success_events.append(kwargs)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self.failure_events.append(kwargs)


@pytest.fixture
def recorder():
    logger = _RecordingLogger()
    saved_flag = litellm.store_batch_line_items_in_callbacks  # test-quality-ok: process-wide opt-in flag; restored in teardown
    saved_success = list(litellm._async_success_callback)  # test-quality-ok: the feature dispatches through this global list; restored in teardown
    saved_failure = list(litellm._async_failure_callback)  # test-quality-ok: same dispatch seam, restored in teardown
    litellm._async_success_callback = [logger]  # test-quality-ok: there is no injection seam for callback lists; teardown restores
    litellm._async_failure_callback = [logger]  # test-quality-ok: same dispatch seam, restored in teardown
    yield logger
    litellm.store_batch_line_items_in_callbacks = saved_flag  # test-quality-ok: teardown restoring the value set above
    litellm._async_success_callback = saved_success  # test-quality-ok: teardown restoring the value set above
    litellm._async_failure_callback = saved_failure  # test-quality-ok: teardown restoring the value set above


def _parent_logging() -> Logging:
    logging_obj = Logging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "<retrieve_batch>"}],
        stream=False,
        call_type="aretrieve_batch",
        start_time=datetime.now(),
        litellm_call_id=str(uuid.uuid4()),
        function_id=str(uuid.uuid4()),
    )
    logging_obj.update_environment_variables(
        litellm_params={"metadata": {"model_info": {"id": "dep-1"}, "model_group": "gpt-4o"}},
        optional_params={},
        custom_llm_provider="openai",
    )
    return logging_obj


async def _log_completed_batch(logging_obj: Logging, batch: LiteLLMBatch) -> None:
    await logging_obj.async_success_handler(
        result=batch,
        batch_cost=1.5,
        batch_usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        batch_models=["gpt-4o"],
        batch_successful_requests=1,
        batch_failed_requests=1,
        batch_prompt_cost=1.0,
        batch_completion_cost=0.5,
    )


def _payload(event: dict) -> dict:
    return event["standard_logging_object"]


def _hidden(event: dict) -> dict:
    return _payload(event)["hidden_params"]


@pytest.mark.asyncio
async def test_line_items_emitted_alongside_aggregate(recorder):
    litellm.store_batch_line_items_in_callbacks = True  # test-quality-ok: the flag under test is a module global; fixture restores it
    batch = _batch()
    with (
        patch("litellm.files.main.afile_content", new_callable=AsyncMock, side_effect=_file_content),  # test-quality-ok: afile_content is the provider boundary; there is no HTTP transport or injection seam for managed file fetch
        patch("litellm.cost_calculator.batch_cost_calculator", return_value=(0.01, 0.02)),  # test-quality-ok: the pricing table boundary, same seam existing batch_utils tests patch
    ):
        await _log_completed_batch(_parent_logging(), batch)

    assert len(recorder.success_events) == 2
    assert len(recorder.failure_events) == 1

    aggregate = next(e for e in recorder.success_events if _hidden(e).get("batch_custom_id") is None)
    assert _payload(aggregate)["response_cost"] == 1.5
    assert "batch_custom_id" not in _hidden(aggregate)

    line = next(e for e in recorder.success_events if _hidden(e).get("batch_custom_id") == "a")
    hidden = _hidden(line)
    assert hidden["batch_id"] == batch.id
    assert hidden["batch_line_status_code"] == 200
    assert line["litellm_params"]["batch_parent_id"] == batch.id

    payload = _payload(line)
    assert payload["response_cost"] == pytest.approx(0.03)
    assert payload["prompt_tokens"] == 10
    assert payload["completion_tokens"] == 5
    assert payload["model_parameters"]["temperature"] == 0.2
    assert any(m.get("content") == "hi a" for m in payload["messages"])
    assert payload["response"]["choices"][0]["message"]["content"] == "hello back"

    failure = recorder.failure_events[0]
    assert _hidden(failure)["batch_custom_id"] == "b"
    assert _hidden(failure)["batch_id"] == batch.id
    assert _hidden(failure)["batch_line_status_code"] == 400
    assert "bad request boom" in _payload(failure)["error_str"]


@pytest.mark.asyncio
async def test_flag_off_emits_only_aggregate(recorder):
    assert litellm.store_batch_line_items_in_callbacks is False
    file_mock = AsyncMock(side_effect=_file_content)
    with patch("litellm.files.main.afile_content", file_mock):  # test-quality-ok: afile_content is the provider boundary; no injection seam for managed file fetch
        await _log_completed_batch(_parent_logging(), _batch())

    assert len(recorder.success_events) == 1
    assert len(recorder.failure_events) == 0
    file_mock.assert_not_called()


@pytest.mark.asyncio
async def test_in_progress_batch_poll_emits_no_line_events(recorder):
    litellm.store_batch_line_items_in_callbacks = True  # test-quality-ok: the flag under test is a module global; fixture restores it
    in_progress: Final = LiteLLMBatch(
        id="batch_wip",
        object="batch",
        endpoint="/v1/chat/completions",
        input_file_id="input-file-1",
        output_file_id=None,
        error_file_id=None,
        status="in_progress",
        completion_window="24h",
        created_at=1,
    )
    file_mock: Final = AsyncMock(side_effect=_file_content)
    with patch("litellm.files.main.afile_content", file_mock):  # test-quality-ok: afile_content is the provider boundary; no injection seam for managed file fetch
        await _parent_logging().async_success_handler(result=in_progress)

    file_mock.assert_not_called()
    assert all(_hidden(e).get("batch_custom_id") is None for e in recorder.success_events)
    assert len(recorder.failure_events) == 0


@pytest.mark.asyncio
async def test_input_fetch_failure_still_emits_aggregate(recorder):
    litellm.store_batch_line_items_in_callbacks = True  # test-quality-ok: the flag under test is a module global; fixture restores it
    with patch("litellm.files.main.afile_content", new_callable=AsyncMock, side_effect=ValueError("boom")):  # test-quality-ok: afile_content is the provider boundary; no injection seam for managed file fetch
        await _log_completed_batch(_parent_logging(), _batch())

    assert len(recorder.success_events) == 1
    assert len(recorder.failure_events) == 0
    assert _payload(recorder.success_events[0])["response_cost"] == 1.5

EDGE_INPUT_JSONL = b"\n".join(
    [
        json.dumps(
            {
                "custom_id": "e",
                "method": "POST",
                "url": "/v1/embeddings",
                "body": {"model": "text-embedding-3-small", "input": "embed me", "encoding_format": "float"},
            }
        ).encode(),
        json.dumps(
            {
                "custom_id": "r",
                "method": "POST",
                "url": "/v1/responses",
                "body": {"model": "gpt-4o", "input": "respond to me"},
            }
        ).encode(),
        json.dumps(
            {
                "custom_id": "badresp",
                "method": "POST",
                "url": "/v1/responses",
                "body": {"model": "gpt-4o", "input": "unreconstructable"},
            }
        ).encode(),
        json.dumps(
            {
                "custom_id": "n",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi n"}]},
            }
        ).encode(),
    ]
)

EDGE_OUTPUT_JSONL = b"\n".join(
    [
        json.dumps(
            {
                "custom_id": "e",
                "response": {
                    "status_code": 200,
                    "body": {
                        "object": "list",
                        "data": [{"object": "embedding", "embedding": [0.1], "index": 0}],
                        "model": "text-embedding-3-small",
                        "usage": {"prompt_tokens": 3, "total_tokens": 3},
                    },
                },
            }
        ).encode(),
        json.dumps(
            {
                "custom_id": "r",
                "response": {
                    "status_code": 200,
                    "body": {
                        "id": "resp_1",
                        "object": "response",
                        "created_at": 1,
                        "status": "completed",
                        "output": [],
                        "model": "gpt-4o",
                    },
                },
            }
        ).encode(),
        json.dumps({"custom_id": "badresp", "response": {"status_code": 200, "body": {}}}).encode(),
        json.dumps({"custom_id": "n", "response": {"body": {"error": {"message": "no status here"}}}}).encode(),
        json.dumps({"custom_id": "boom", "response": "not-a-dict"}).encode(),
        json.dumps(
            {
                "custom_id": "mi",
                "modelInput": {"messages": [{"role": "user", "content": "hi mi"}]},
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "mi back"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
                    },
                },
            }
        ).encode(),
    ]
)

ANTHROPIC_INPUT_JSONL = json.dumps(
    {
        "custom_id": "b2",
        "params": {"model": "claude-3", "max_tokens": 5, "messages": [{"role": "user", "content": "hi b2"}]},
    }
).encode()

ANTHROPIC_OUTPUT_JSONL = json.dumps(
    {
        "custom_id": "b2",
        "result": {
            "type": "succeeded",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "hello b2"}],
                "model": "claude-3",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        },
    }
).encode()


def _edge_file_content(file_id: str, **_kwargs):
    return SimpleNamespace(
        content={
            "input-2": EDGE_INPUT_JSONL,
            "output-2": EDGE_OUTPUT_JSONL,
            "input-anth": ANTHROPIC_INPUT_JSONL,
            "output-anth": ANTHROPIC_OUTPUT_JSONL,
        }[file_id]
    )


def _parent_logging_with_params(litellm_params: dict) -> Logging:
    logging_obj = _parent_logging()
    logging_obj.update_environment_variables(
        litellm_params=litellm_params,
        optional_params={},
        custom_llm_provider="openai",
    )
    return logging_obj


@pytest.mark.asyncio
async def test_line_items_edge_shapes_and_edge_cases(recorder):
    litellm.store_batch_line_items_in_callbacks = True  # test-quality-ok: the flag under test is a module global; fixture restores it
    batch = LiteLLMBatch(
        id="batch_edge",
        object="batch",
        endpoint="/v1/chat/completions",
        input_file_id="input-2",
        output_file_id="output-2",
        error_file_id=None,
        status="completed",
        completion_window="24h",
        created_at=1,
    )
    file_mock: Final = AsyncMock(side_effect=_edge_file_content)
    parent: Final = _parent_logging_with_params(
        {
            "metadata": {"model_info": {"id": "dep-1"}, "model_group": "gpt-4o"},
        }
    )
    parent._litellm_internal_model_credentials = {"api_key": "sk-line-items-marker"}  # test-quality-ok: private transport attribute, same channel the batch cost tracker uses
    with (
        patch("litellm.files.main.afile_content", file_mock),  # test-quality-ok: afile_content is the provider boundary; no injection seam for managed file fetch
        patch("litellm.cost_calculator.batch_cost_calculator", return_value=(0.01, 0.02)),  # test-quality-ok: the pricing table boundary, same seam existing batch_utils tests patch
    ):
        await _log_completed_batch(parent, batch)

    by_custom_id = {_hidden(e).get("batch_custom_id"): e for e in recorder.success_events}
    aggregate = next(e for e in recorder.success_events if _hidden(e).get("batch_custom_id") is None)
    assert _payload(aggregate)["response_cost"] == 1.5

    assert by_custom_id["e"]["litellm_params"]["batch_parent_id"] == batch.id
    assert by_custom_id["e"]["call_type"] == "aembedding"
    assert _payload(by_custom_id["e"])["response"]["data"][0]["embedding"] == [0.1]

    assert by_custom_id["r"]["call_type"] == "aresponses"
    assert _payload(by_custom_id["r"])["response"]["id"] == "resp_1"

    assert by_custom_id["mi"]["call_type"] == "acompletion"
    assert _hidden(by_custom_id["mi"])["batch_line_status_code"] == 200

    assert "badresp" not in by_custom_id
    assert "boom" not in by_custom_id

    failure = recorder.failure_events[0]
    assert _hidden(failure)["batch_custom_id"] == "n"
    assert _hidden(failure)["batch_line_status_code"] is None
    assert "no status here" in _payload(failure)["error_str"]

    assert "sk-line-items-marker" in str(file_mock.call_args_list)
    assert "sk-line-items-marker" not in str(by_custom_id["e"]["litellm_params"])
    file_ids_fetched = [call.kwargs.get("file_id") or call.args[0] for call in file_mock.call_args_list]
    assert "input-2" in file_ids_fetched and "output-2" in file_ids_fetched
    assert not any(file_id is None for file_id in file_ids_fetched)


@pytest.mark.asyncio
async def test_line_items_anthropic_shapes(recorder):
    litellm.store_batch_line_items_in_callbacks = True  # test-quality-ok: the flag under test is a module global; fixture restores it
    batch = LiteLLMBatch(
        id="batch_anth",
        object="batch",
        endpoint="/v1/messages",
        input_file_id="input-anth",
        output_file_id="output-anth",
        error_file_id=None,
        status="completed",
        completion_window="24h",
        created_at=1,
    )
    file_mock: Final = AsyncMock(side_effect=_edge_file_content)
    logging_obj = _parent_logging()
    logging_obj.update_environment_variables(
        litellm_params={"metadata": {"model_info": {"id": "dep-1"}, "model_group": "claude-3"}},
        optional_params={},
        custom_llm_provider="anthropic",
    )
    with (
        patch("litellm.files.main.afile_content", file_mock),  # test-quality-ok: afile_content is the provider boundary; no injection seam for managed file fetch
        patch("litellm.cost_calculator.batch_cost_calculator", return_value=(0.01, 0.02)),  # test-quality-ok: the pricing table boundary, same seam existing batch_utils tests patch
    ):
        await _log_completed_batch(logging_obj, batch)

    line = next(e for e in recorder.success_events if _hidden(e).get("batch_custom_id") == "b2")
    assert _hidden(line)["batch_line_status_code"] == 200
    assert line["litellm_params"]["batch_parent_id"] == batch.id
