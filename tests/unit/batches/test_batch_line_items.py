import json
import logging
from hashlib import sha256

import pytest

import litellm
from litellm.batches.batch_line_items import (
    BatchLineItemSource,
    batch_line_item_parts,
    dispatch_batch_line_items,
    line_item_metadata,
    without_request_and_response,
)
from litellm.batches.batch_utils import calculate_batch_cost_and_usage, get_file_content_as_dictionary
from litellm.constants import BATCH_LINE_ITEMS_MAX_PART_BYTES
from litellm.integrations.batch_line_item_objects import batch_line_items_ndjson, batch_line_items_object_name
from litellm.integrations.custom_logger import CustomLogger
from litellm.llms.vertex_ai.files.transformation import VertexAIFilesConfig, _OpenAIToVertexBatchUploadStream
from litellm.types.utils import ModelInfo, StandardBatchLineItemsPayload

INPUT_RATE = 0.001
OUTPUT_RATE = 0.002
FINISHED_AT = 1791494130
ALICE_KEY_HASH = sha256(b"sk-alice").hexdigest()


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
            {"user_api_key_hash": ALICE_KEY_HASH, "user_api_key_team_id": "team-alpha", "tags": ["env:prod"]}
        ),
    }
    return BatchLineItemSource(**{**fields, **overrides})


def _whole_batch(source: BatchLineItemSource, input_content: bytes, result_contents: tuple[bytes, ...]):
    """Every line of a batch small enough for one part."""
    (payload,) = batch_line_item_parts(source, input_content, result_contents, BATCH_LINE_ITEMS_MAX_PART_BYTES)
    return payload


def _payload() -> StandardBatchLineItemsPayload:
    return _whole_batch(
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
    payload = _whole_batch(
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
            "user_api_key_hash": ALICE_KEY_HASH,
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

    payload = _whole_batch(_source(), b"", (outputs,))
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

    payload = _whole_batch(
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

    payload = _whole_batch(
        _source(custom_llm_provider="bedrock", cost_provider="bedrock"), _jsonl(request), (_jsonl(result),)
    )

    (line,) = payload["lines"]
    assert (line["custom_id"], line["status"], line["request"]) == ("r-1", "success", request)


def test_vertex_lines_pair_with_the_input_row_that_carries_their_custom_id():
    vertex_input = b"\n".join(
        _OpenAIToVertexBatchUploadStream(
            _jsonl(_request("vx-0", "first"), _request("vx-1", "second")),
            VertexAIFilesConfig()._map_openai_to_vertex_params,
        ).iter_bytes()
    )
    vertex_rows = get_file_content_as_dictionary(vertex_input)
    assert all("custom_id" not in row for row in vertex_rows)

    payload = _whole_batch(
        _source(model="gemini-2.5-flash", custom_llm_provider="vertex_ai", cost_provider="vertex_ai"),
        vertex_input,
        (_jsonl(_success("vx-1", 4, 2), _success("vx-0", 3, 1)),),
    )

    assert [(line["custom_id"], line["request"]) for line in payload["lines"]] == [
        ("vx-1", vertex_rows[1]),
        ("vx-0", vertex_rows[0]),
    ]


def test_a_vertex_embeddings_entry_split_into_several_rows_pairs_with_none_of_them():
    embeddings = {
        "custom_id": "emb-1",
        "method": "POST",
        "url": "/v1/embeddings",
        "body": {"model": "gemini-embedding-001", "input": ["first", "second"]},
    }
    vertex_input = b"\n".join(
        _OpenAIToVertexBatchUploadStream(
            _jsonl(embeddings, _request("vx-0", "chat")), VertexAIFilesConfig()._map_openai_to_vertex_params
        ).iter_bytes()
    )
    vertex_rows = get_file_content_as_dictionary(vertex_input)
    assert [row.get("key") for row in vertex_rows] == ["emb-1#0/2", "emb-1#1/2", None]

    payload = _whole_batch(
        _source(model="gemini-2.5-flash", custom_llm_provider="vertex_ai", cost_provider="vertex_ai"),
        vertex_input,
        (_jsonl(_success("emb-1", 2, 0), _success("vx-0", 3, 1)),),
    )

    assert [(line["custom_id"], line["request"]) for line in payload["lines"]] == [
        ("emb-1", None),
        ("vx-0", vertex_rows[2]),
    ]


def test_a_vertex_input_row_without_a_custom_id_pairs_with_nothing():
    unlabeled = {"request": {"contents": [{"role": "user", "parts": [{"text": "hi"}]}]}}

    payload = _whole_batch(
        _source(model="gemini-2.5-flash", custom_llm_provider="vertex_ai", cost_provider="vertex_ai"),
        _jsonl(unlabeled),
        (_jsonl({**_success("x", 3, 1), "custom_id": None}),),
    )

    (line,) = payload["lines"]
    assert (line["custom_id"], line["request"]) == (None, None)


@pytest.mark.parametrize(
    "vertex_input",
    [
        pytest.param(
            b"\n".join(
                _OpenAIToVertexBatchUploadStream(
                    _jsonl(_request("unknown", "first"), {**_request("vx-1", "second"), "custom_id": None}),
                    VertexAIFilesConfig()._map_openai_to_vertex_params,
                ).iter_bytes()
            ),
            id="custom-id-unknown-next-to-a-null-one",
        ),
        pytest.param(
            _jsonl(
                {"request": {"contents": [{"role": "user", "parts": [{"text": "first"}]}], "labels": {"team": "a"}}},
                {"request": {"contents": [{"role": "user", "parts": [{"text": "second"}]}]}},
            ),
            id="native-rows-with-other-labels-next-to-unlabeled-ones",
        ),
    ],
)
def test_a_vertex_line_read_as_unknown_pairs_with_no_input_row(vertex_input):
    payload = _whole_batch(
        _source(model="gemini-2.5-flash", custom_llm_provider="vertex_ai", cost_provider="vertex_ai"),
        vertex_input,
        (_jsonl(_success("unknown", 3, 1), _success("unknown", 4, 2)),),
    )

    assert [(line["custom_id"], line["request"]) for line in payload["lines"]] == [("unknown", None)] * 2


@pytest.mark.asyncio
async def test_redaction_drops_only_the_request_and_response():
    payload = _payload()

    redacted = without_request_and_response(payload)

    assert redacted["batch_id"] == payload["batch_id"]
    assert redacted["finished_at"] == payload["finished_at"]
    assert redacted["part"] == payload["part"]
    assert [{**line, "request": None, "response": None} for line in payload["lines"]] == list(redacted["lines"])
    assert all(line["request"] is not None for line in payload["lines"])


@pytest.mark.asyncio
async def test_the_object_holds_one_json_line_per_batch_line_under_its_finished_day():
    payload = _payload()

    ndjson = batch_line_items_ndjson(payload)

    assert [json.loads(line) for line in ndjson.splitlines()] == list(payload["lines"])
    assert ndjson.endswith("}\n")
    assert (
        batch_line_items_object_name(payload)
        == "batch_line_items/2026-10-08/bGl0ZWxsbV9wcm94eTtiYXRjaA/part-00000.jsonl"
    )
    assert (
        batch_line_items_object_name(
            {**payload, "batch_id": "projects/p/locations/l/batchPredictionJobs/9", "part": 12}
        )
        == "batch_line_items/2026-10-08/projects_p_locations_l_batchPredictionJobs_9/part-00012.jsonl"
    )


def _raw_size(request: dict, result: dict) -> int:
    return len(json.dumps(request)) + len(json.dumps(result))


@pytest.mark.parametrize(
    "extra_bytes, parts",
    [(0, [["a", "b"], ["c"]]), (1, [["a", "b", "c"]])],
    ids=["closes once its raw lines reach the size", "stays open below the size"],
)
def test_a_part_closes_once_its_raw_request_and_result_lines_reach_the_part_size(extra_bytes, parts):
    requests = [_request(custom_id, f"question {custom_id}") for custom_id in "abc"]
    results = [_success(custom_id, 10, 5) for custom_id in "abc"]
    max_part_bytes = _raw_size(requests[0], results[0]) + _raw_size(requests[1], results[1]) + extra_bytes

    payloads = list(batch_line_item_parts(_source(), _jsonl(*requests), (_jsonl(*results),), max_part_bytes))

    assert [payload["part"] for payload in payloads] == list(range(len(parts)))
    assert [[line["custom_id"] for line in payload["lines"]] for payload in payloads] == parts
    assert [line["request"] for payload in payloads for line in payload["lines"]] == requests


def test_blank_and_unreadable_input_lines_leave_the_other_lines_paired():
    input_content = b"\n".join(
        [json.dumps(_request("a", "first")).encode(), b"", b"not json", json.dumps(_request("b", "second")).encode()]
    )

    payload = _whole_batch(_source(), input_content, (_jsonl(_success("a", 1, 1), _success("b", 1, 1)),))

    assert [line["request"] for line in payload["lines"]] == [_request("a", "first"), _request("b", "second")]


def test_files_with_no_readable_line_make_no_part():
    assert list(batch_line_item_parts(_source(), b"", (b"not json\n", b""), BATCH_LINE_ITEMS_MAX_PART_BYTES)) == []


@pytest.mark.parametrize(
    "saved, recorded",
    [
        (ALICE_KEY_HASH, ALICE_KEY_HASH),
        ("ya29.opaque-oauth2-access-token", sha256(b"ya29.opaque-oauth2-access-token").hexdigest()),
        ("litellm_proxy_master_key", sha256(b"litellm_proxy_master_key").hexdigest()),
        (None, None),
    ],
    ids=["virtual key hash", "oauth2 token", "master key alias", "nothing saved"],
)
def test_the_creator_credential_is_recorded_only_as_a_hash(saved, recorded):
    assert line_item_metadata({"user_api_key_hash": saved})["user_api_key_hash"] == recorded


def test_a_line_holding_a_lone_surrogate_is_escaped_so_the_object_still_encodes_as_utf8():
    payload = _payload()
    (line, *_) = payload["lines"]
    surrogate_line = {**line, "custom_id": "emoji-\ud83d", "response": {"body": {"content": "caf\u00e9 \ud83d"}}}
    readable_line = {**line, "custom_id": "caf\u00e9"}

    ndjson = batch_line_items_ndjson({**payload, "lines": (surrogate_line, readable_line)})

    surrogate_text, readable_text = ndjson.encode("utf-8").decode("utf-8").splitlines()
    assert json.loads(surrogate_text) == surrogate_line
    assert "\\ud83d" in surrogate_text
    assert json.loads(readable_text) == readable_line
    assert '"custom_id": "caf\u00e9"' in readable_text


@pytest.mark.asyncio
async def test_a_logger_gets_the_request_and_response_unless_message_logging_is_off(monkeypatch):
    payload = _payload()
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
    payload = _payload()
    logger = _RecordingLogger()
    monkeypatch.setattr(litellm, "batch_line_item_callbacks", (logger,))

    await dispatch_batch_line_items(payload, redact_bodies=True)

    assert logger.received == [without_request_and_response(payload)]


@pytest.mark.asyncio
async def test_a_failing_or_unknown_sink_is_logged_and_the_other_sinks_still_get_the_lines(monkeypatch, caplog):
    payload = _payload()
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
