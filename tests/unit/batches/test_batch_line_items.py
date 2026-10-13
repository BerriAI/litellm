import json
import logging

import pytest

import litellm
from litellm.batches.batch_line_items import (
    BatchLineItemSource,
    build_batch_line_items,
    dispatch_batch_line_items,
    line_item_metadata,
    without_request_and_response,
)
from litellm.batches.batch_utils import calculate_batch_cost_and_usage, get_file_content_as_dictionary
from litellm.integrations.batch_line_item_objects import batch_line_items_ndjson, batch_line_items_object_name
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelInfo, StandardBatchLineItemsPayload

INPUT_RATE = 0.001
OUTPUT_RATE = 0.002
FINISHED_AT = 1791494130


def _jsonl(*lines: dict) -> bytes:
    return "".join(json.dumps(line) + "\n" for line in lines).encode()


def _request(custom_id: str, text: str) -> dict:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": text}]},
    }


def _success(custom_id: str, prompt_tokens: int, completion_tokens: int) -> dict:
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": {
            "status_code": 200,
            "body": {
                "model": "gpt-4o-mini",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            },
        },
        "error": None,
    }


def _error(custom_id: str) -> dict:
    return {
        "id": f"batch_req_{custom_id}",
        "custom_id": custom_id,
        "response": {"status_code": 400, "body": {"error": {"message": "max_tokens is too large"}}},
        "error": None,
    }


def _source(**overrides) -> BatchLineItemSource:
    fields = {
        "batch_id": "bGl0ZWxsbV9wcm94eTtiYXRjaA",
        "finished_at": FINISHED_AT,
        "model": "gpt-4o-mini",
        "custom_llm_provider": "openai",
        "cost_provider": "openai",
        "model_info": ModelInfo(input_cost_per_token_batches=INPUT_RATE, output_cost_per_token_batches=OUTPUT_RATE),
        "metadata": line_item_metadata(
            {"user_api_key_hash": "hash-alice", "user_api_key_team_id": "team-alpha", "tags": ["env:prod"]}
        ),
    }
    return BatchLineItemSource(**{**fields, **overrides})


async def _payload() -> StandardBatchLineItemsPayload:
    return await build_batch_line_items(
        _source(),
        _jsonl(_request("ok-1", "first question"), _request("bad-1", "second question")),
        (_jsonl(_success("ok-1", 100, 50)), _jsonl(_error("bad-1"))),
    )


class _RecordingLogger(CustomLogger):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.received: list[StandardBatchLineItemsPayload] = []

    async def async_log_batch_line_items(self, payload: StandardBatchLineItemsPayload) -> None:
        self.received.append(payload)


class _FailingLogger(CustomLogger):
    async def async_log_batch_line_items(self, payload: StandardBatchLineItemsPayload) -> None:
        raise RuntimeError("bucket is gone")


@pytest.mark.asyncio
async def test_each_result_line_is_paired_with_the_request_that_shares_its_custom_id():
    payload = await build_batch_line_items(
        _source(),
        _jsonl(_request("ok-1", "first"), _request("bad-1", "second"), _request("never-ran", "third")),
        (_jsonl(_success("ok-1", 100, 50)), _jsonl(_error("bad-1"))),
    )

    assert payload["batch_id"] == "bGl0ZWxsbV9wcm94eTtiYXRjaA"
    assert payload["finished_at"] == FINISHED_AT
    succeeded, failed = payload["lines"]
    assert succeeded == {
        "batch_id": "bGl0ZWxsbV9wcm94eTtiYXRjaA",
        "custom_id": "ok-1",
        "status": "success",
        "request": _request("ok-1", "first"),
        "response": _success("ok-1", 100, 50),
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "response_cost": pytest.approx(100 * INPUT_RATE + 50 * OUTPUT_RATE),
        "model": "gpt-4o-mini",
        "custom_llm_provider": "openai",
        "metadata": {
            "user_api_key_hash": "hash-alice",
            "user_api_key_alias": None,
            "user_api_key_team_id": "team-alpha",
            "user_api_key_team_alias": None,
            "user_api_key_user_id": None,
            "user_api_key_org_id": None,
        },
    }
    assert failed["custom_id"] == "bad-1"
    assert failed["status"] == "failure"
    assert failed["request"] == _request("bad-1", "second")
    assert failed["response"] == _error("bad-1")
    assert failed["usage"] is None
    assert failed["response_cost"] == 0


@pytest.mark.asyncio
async def test_line_costs_add_up_to_the_batch_cost_the_spend_row_records():
    outputs = _jsonl(_success("a", 100, 50), _success("b", 7, 3), _error("c"))

    payload = await build_batch_line_items(_source(), b"", (outputs,))
    aggregate = await calculate_batch_cost_and_usage(
        get_file_content_as_dictionary(outputs), "openai", model_name="gpt-4o-mini", model_info=_source().model_info
    )

    assert [line["response_cost"] for line in payload["lines"]] == pytest.approx(
        [100 * INPUT_RATE + 50 * OUTPUT_RATE, 7 * INPUT_RATE + 3 * OUTPUT_RATE, 0]
    )
    assert sum(line["response_cost"] for line in payload["lines"]) == pytest.approx(aggregate.cost)


@pytest.mark.asyncio
async def test_anthropic_results_without_an_input_file_keep_their_status_and_usage():
    succeeded = {
        "custom_id": "a-1",
        "result": {
            "type": "succeeded",
            "message": {"model": "claude-sonnet-5-5", "usage": {"input_tokens": 10, "output_tokens": 4}},
        },
    }
    errored = {"custom_id": "a-2", "result": {"type": "errored", "error": {"type": "invalid_request_error"}}}

    payload = await build_batch_line_items(
        _source(model="claude-sonnet-5-5", custom_llm_provider="anthropic", cost_provider="anthropic"),
        b"",
        (_jsonl(succeeded, errored),),
    )

    first, second = payload["lines"]
    assert (first["custom_id"], first["status"], first["request"]) == ("a-1", "success", None)
    assert first["usage"] == {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
    assert (second["custom_id"], second["status"], second["usage"]) == ("a-2", "failure", None)


@pytest.mark.asyncio
async def test_bedrock_lines_pair_by_record_id():
    request = {"recordId": "r-1", "modelInput": {"messages": [{"role": "user", "content": "hi"}]}}
    result = {**request, "modelOutput": {"usage": {"input_tokens": 3, "output_tokens": 2}}}

    payload = await build_batch_line_items(
        _source(custom_llm_provider="bedrock", cost_provider="bedrock"), _jsonl(request), (_jsonl(result),)
    )

    (line,) = payload["lines"]
    assert (line["custom_id"], line["status"], line["request"]) == ("r-1", "success", request)


@pytest.mark.asyncio
async def test_redaction_drops_only_the_request_and_response():
    payload = await _payload()

    redacted = without_request_and_response(payload)

    assert redacted["batch_id"] == payload["batch_id"]
    assert redacted["finished_at"] == payload["finished_at"]
    assert [{**line, "request": None, "response": None} for line in payload["lines"]] == list(redacted["lines"])
    assert all(line["request"] is not None for line in payload["lines"])


@pytest.mark.asyncio
async def test_the_object_holds_one_json_line_per_batch_line_under_its_finished_day():
    payload = await _payload()

    ndjson = batch_line_items_ndjson(payload)

    assert [json.loads(line) for line in ndjson.splitlines()] == list(payload["lines"])
    assert ndjson.endswith("}\n")
    assert batch_line_items_object_name(payload) == "batch_line_items/2026-10-08/bGl0ZWxsbV9wcm94eTtiYXRjaA.jsonl"
    assert (
        batch_line_items_object_name({**payload, "batch_id": "projects/p/locations/l/batchPredictionJobs/9"})
        == "batch_line_items/2026-10-08/projects_p_locations_l_batchPredictionJobs_9.jsonl"
    )


@pytest.mark.asyncio
async def test_a_logger_gets_the_request_and_response_unless_message_logging_is_off(monkeypatch):
    payload = await _payload()
    open_logger, quiet_logger, legacy_logger = (
        _RecordingLogger(),
        _RecordingLogger(turn_off_message_logging=True),
        _RecordingLogger(message_logging=False),
    )
    monkeypatch.setattr(litellm, "batch_line_item_callbacks", (open_logger, quiet_logger, legacy_logger))

    await dispatch_batch_line_items(payload, redact_bodies=False)

    assert open_logger.received == [payload]
    assert quiet_logger.received == [without_request_and_response(payload)]
    assert legacy_logger.received == [without_request_and_response(payload)]


@pytest.mark.asyncio
async def test_redacted_dispatch_never_hands_a_logger_the_bodies(monkeypatch):
    payload = await _payload()
    logger = _RecordingLogger()
    monkeypatch.setattr(litellm, "batch_line_item_callbacks", (logger,))

    await dispatch_batch_line_items(payload, redact_bodies=True)

    assert logger.received == [without_request_and_response(payload)]


@pytest.mark.asyncio
async def test_a_failing_or_unknown_sink_is_logged_and_the_other_sinks_still_get_the_lines(monkeypatch, caplog):
    payload = await _payload()
    logger = _RecordingLogger()
    monkeypatch.setattr(litellm, "batch_line_item_callbacks", (_FailingLogger(), "langfuse", logger))

    with caplog.at_level(logging.ERROR, logger="LiteLLM"):
        await dispatch_batch_line_items(payload, redact_bodies=False)

    assert logger.received == [payload]
    failures = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    assert len(failures) == 2
    assert all(payload["batch_id"] in message for message in failures)
    assert any("bucket is gone" in message for message in failures)
    assert any("'langfuse'" in message for message in failures)
