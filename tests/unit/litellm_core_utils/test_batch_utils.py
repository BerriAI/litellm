import asyncio
import importlib
import json
from io import BytesIO
from typing import Dict, Final, List

import pytest
from openai.types.batch import BatchRequestCounts

import litellm
from litellm.litellm_core_utils.batch_utils import (
    POLLED_BATCH_COST_IS_FINAL_KWARG,
    POLLED_BATCH_ID_KWARG,
    ChargeOnceJob,
    InMemoryFile,
    batch_cost_is_final,
    defers_batch_cost,
    get_router_metadata_variable_name,
    parse_jsonl_with_embedded_newlines,
    replace_model_in_jsonl,
)
from litellm.types.utils import LiteLLMBatch
from litellm.types.videos.utils import POLLED_VIDEO_STATUS_KWARG, encode_video_id_with_provider
from tests._vcr_conftest_common import install_live_call_probe, record_vcr_outcome


# Fixtures
@pytest.fixture
def sample_jsonl_data() -> List[Dict]:
    """Fixture providing sample JSONL data"""
    return [
        {
            "body": {
                "model": "gpt-5-mini",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        },
        {"body": {"model": "gpt-5.5", "messages": [{"role": "user", "content": "Hi"}]}},
    ]


@pytest.fixture
def sample_jsonl_bytes(sample_jsonl_data) -> bytes:
    """Fixture providing sample JSONL as bytes"""
    jsonl_str = "\n".join(json.dumps(line) for line in sample_jsonl_data)
    return jsonl_str.encode("utf-8")


@pytest.fixture
def sample_file_like(sample_jsonl_bytes):
    """Fixture providing a file-like object"""
    return BytesIO(sample_jsonl_bytes)


# Test cases
def test_bytes_input(sample_jsonl_bytes):
    """Test with bytes input"""
    new_model = "claude-3"
    result = replace_model_in_jsonl(sample_jsonl_bytes, new_model)

    assert result is not None
    assert isinstance(result, InMemoryFile)
    assert result.name == "modified_file.jsonl"
    assert result.content_type == "application/jsonl"


def test_tuple_input(sample_jsonl_bytes):
    """Test with tuple input"""
    new_model = "claude-3"
    test_tuple = ("test.jsonl", sample_jsonl_bytes, "application/json")
    result = replace_model_in_jsonl(test_tuple, new_model)

    assert result is not None
    assert isinstance(result, InMemoryFile)
    assert result.name == "modified_file.jsonl"
    assert result.content_type == "application/jsonl"


def test_tuple_with_file_handle_rewrites_model(sample_jsonl_bytes):
    """Security regression: when the tuple's content element is a file handle
    (batch uploads stream from the spooled upload handle), the model must still
    be rewritten. Otherwise a restricted body.model survives unmodified and
    bypasses the batch model allowlist, which only checks the upload target."""
    new_model = "approved-target-model"
    handle = BytesIO(sample_jsonl_bytes)
    test_tuple = ("test.jsonl", handle, "application/json")

    result = replace_model_in_jsonl(test_tuple, new_model)

    assert isinstance(result, InMemoryFile)
    rows = [json.loads(line) for line in result.getvalue().decode("utf-8").splitlines() if line.strip()]
    assert rows, "rewrite must produce rows"
    # every row now carries the rewritten target, not the original (restricted) model
    assert all(row["body"]["model"] == new_model for row in rows)
    assert all(row["body"]["model"] != "gpt-5.5" for row in rows)


def test_file_like_object(sample_file_like):
    """Test with file-like object input"""
    new_model = "claude-3"
    result = replace_model_in_jsonl(sample_file_like, new_model)

    assert result is not None
    assert isinstance(result, InMemoryFile)
    assert result.name == "modified_file.jsonl"
    assert result.content_type == "application/jsonl"


def test_router_metadata_variable_name():
    """Test that the variable name is correct"""
    assert get_router_metadata_variable_name(function_name="completion") == "metadata"
    assert (
        get_router_metadata_variable_name(function_name="batch") == "litellm_metadata"
    )
    assert (
        get_router_metadata_variable_name(function_name="acreate_file")
        == "litellm_metadata"
    )
    assert (
        get_router_metadata_variable_name(function_name="aget_file")
        == "litellm_metadata"
    )


def test_non_json_input():
    """Test that replace_model_in_jsonl returns original content for non-JSON input"""
    from litellm.litellm_core_utils.batch_utils import replace_model_in_jsonl

    # Test with non-JSON string
    non_json_str = "This is not a JSON string"
    result = replace_model_in_jsonl(non_json_str, "gpt-4")
    assert result == non_json_str

    # Test with non-JSON bytes
    non_json_bytes = b"This is not JSON bytes"
    result = replace_model_in_jsonl(non_json_bytes, "gpt-4")
    assert result == non_json_bytes

    # Test with non-JSON file-like object
    from io import BytesIO

    non_json_file = BytesIO(b"This is not JSON in a file")
    result = replace_model_in_jsonl(non_json_file, "gpt-4")
    assert result == non_json_file


def test_should_replace_model_in_jsonl():
    """Test that should_replace_model_in_jsonl returns the correct value"""
    from litellm.litellm_core_utils.batch_utils import should_replace_model_in_jsonl

    assert should_replace_model_in_jsonl(purpose="batch") is True
    assert should_replace_model_in_jsonl(purpose="batch", passthrough=True) is False
    assert should_replace_model_in_jsonl(purpose="test") is False
    assert should_replace_model_in_jsonl(purpose="user_data") is False


def test_parse_jsonl_with_embedded_newlines_simple():
    """Test parsing simple JSONL without embedded newlines"""
    content = '{"id": 1, "name": "test"}\n{"id": 2, "name": "test2"}'
    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 2
    assert result[0] == {"id": 1, "name": "test"}
    assert result[1] == {"id": 2, "name": "test2"}


def test_parse_jsonl_with_embedded_newlines_in_strings():
    """Test parsing JSONL with newlines embedded in string values"""
    content = '{"id": 1, "message": "Line 1\\nLine 2\\nLine 3"}\n{"id": 2, "message": "test"}'
    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 2
    assert result[0] == {"id": 1, "message": "Line 1\nLine 2\nLine 3"}
    assert result[1] == {"id": 2, "message": "test"}


def test_parse_jsonl_with_embedded_newlines_real_world_example():
    """Test with the real-world example from the Cooler Master Shark X case"""
    # This simulates the actual problem case from the user's log
    content = """{"custom_id":"16546277850245725","method":"POST","url":"/v1/chat/completions","body":{"model":"openai-gpt-4o-mini-dp-items-translation-dag","messages":[{"role":"system","content":"Translate the product title and description for an e-commerce marketplace in Saudi Arabia and the UAE. Text may be in English or Arabic.\\n"},{"role":"user","content":"\\nOriginal Title: ```Cooler Master Shark X PC Case```\\nOriginal Description: ```UNIQUE MASTERPIECEShark X is a system that provides an impressive  unique alternative to traditional PC systems.  Shark X will stand out and can be the ultimate  trophy or conversation piece for people looking  for a unique setup that stands head and fins  above the res.```\\nStore Name: ```geekay```\\n"}]}}"""

    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 1
    assert result[0]["custom_id"] == "16546277850245725"
    assert result[0]["method"] == "POST"
    assert result[0]["body"]["model"] == "openai-gpt-4o-mini-dp-items-translation-dag"
    assert len(result[0]["body"]["messages"]) == 2
    assert "Translate the product title" in result[0]["body"]["messages"][0]["content"]
    assert "Cooler Master Shark X PC Case" in result[0]["body"]["messages"][1]["content"]
    assert "UNIQUE MASTERPIECEShark X" in result[0]["body"]["messages"][1]["content"]


def test_parse_jsonl_with_embedded_newlines_multiple_complex_objects():
    """Test parsing multiple complex JSON objects with embedded newlines"""
    content = """{"id":1,"text":"Line 1\\nLine 2"}
{"id":2,"nested":{"field":"Value\\nWith\\nNewlines"}}
{"id":3,"simple":"test"}"""

    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 3
    assert result[0]["id"] == 1
    assert result[0]["text"] == "Line 1\nLine 2"
    assert result[1]["id"] == 2
    assert result[1]["nested"]["field"] == "Value\nWith\nNewlines"
    assert result[2]["id"] == 3
    assert result[2]["simple"] == "test"


def test_parse_jsonl_with_embedded_newlines_no_trailing_newline():
    """Test parsing JSONL without trailing newline"""
    content = '{"id": 1, "name": "test"}'
    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 1
    assert result[0] == {"id": 1, "name": "test"}


def test_parse_jsonl_with_embedded_newlines_empty_string():
    """Test parsing empty string"""
    content = ""
    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 0


def test_parse_jsonl_with_embedded_newlines_whitespace_only():
    """Test parsing whitespace-only content"""
    content = "   \n  \n  "
    result = parse_jsonl_with_embedded_newlines(content)

    assert len(result) == 0


def test_replace_model_in_jsonl_malformed_middle_row_returns_original():
    """Regression: a malformed/truncated middle row must not silently drop the
    rows that follow it. The streaming rewrite accumulates physical lines into a
    buffer; a row that never parses poisons the buffer so every later valid row
    is concatenated into it and dropped. Returning that partial rewrite would
    ship a truncated batch with no error to the caller. Instead the original
    content is returned unchanged so the provider rejects the bad batch loudly."""
    content = (
        b'{"custom_id":"a","body":{"model":"x"}}\n'
        b'{"custom_id":"b","body":{"model":\n'  # truncated, never completes
        b'{"custom_id":"c","body":{"model":"x"}}\n'
    )

    result = replace_model_in_jsonl(content, "new-model")

    assert result == content, "must return the original unchanged, not a partial rewrite"


def test_replace_model_in_jsonl_malformed_row_seekable_handle_rewound():
    """When the source is a seekable handle that gets consumed during the failed
    rewrite, it must be rewound to 0 so the caller can re-read the full original."""
    content = (
        b'{"custom_id":"a","body":{"model":"x"}}\n'
        b'{"custom_id":"b","body":{"model":\n'
        b'{"custom_id":"c","body":{"model":"x"}}\n'
    )
    handle = BytesIO(content)

    result = replace_model_in_jsonl(handle, "new-model")

    assert result is handle
    assert handle.read() == content, "handle must be rewound for the caller to re-read"


def test_replace_model_in_jsonl_multi_row_rewrites_every_model():
    """Happy path: a well-formed multi-row file gets every row's model rewritten
    and no row is dropped."""
    content = (
        b'{"custom_id":"a","body":{"model":"old1"}}\n'
        b'{"custom_id":"b","body":{"model":"old2"}}\n'
        b'{"custom_id":"c","body":{"model":"old3"}}\n'
    )

    result = replace_model_in_jsonl(content, "new-model")

    assert isinstance(result, InMemoryFile)
    rows = [json.loads(line) for line in result.getvalue().decode("utf-8").splitlines() if line.strip()]
    assert [row["custom_id"] for row in rows] == ["a", "b", "c"]
    assert all(row["body"]["model"] == "new-model" for row in rows)


def test_replace_model_in_jsonl_keeps_every_row_on_its_original_line():
    content = (
        b'{"custom_id":"a","body":{"model":"old"}}\n'
        b"\n"
        b'{"custom_id":"c","body":{"model":"old"}}\n'
        b'{"custom_id":"d",\n'
        b'"body":{"model":"old"}}\n'
        b'{"custom_id":"f","body":{"model":"old"}}\n'
    )

    result = replace_model_in_jsonl(content, "new-model", keep_line_numbers=True)

    assert isinstance(result, InMemoryFile)
    lines = result.getvalue().decode("utf-8").split("\n")
    assert [json.loads(line)["custom_id"] if line else None for line in lines] == ["a", None, "c", None, "d", "f"]
    assert all(json.loads(line)["body"]["model"] == "new-model" for line in lines if line)


@pytest.mark.parametrize(
    "content",
    [
        b'\xef\xbb\xbf{"custom_id":"a","body":{"model":"old"}}\n\xef\xbb\xbf{"custom_id":"b","body":{"model":"old"}}\n',
        '\ufeff{"custom_id":"a","body":{"model":"old"}}\n\ufeff{"custom_id":"b","body":{"model":"old"}}\n',
    ],
    ids=["bytes", "str"],
)
def test_replace_model_in_jsonl_rewrites_rows_that_start_with_a_bom(content):
    result = replace_model_in_jsonl(content, "new-model")

    assert isinstance(result, InMemoryFile)
    rows = [json.loads(line) for line in result.getvalue().decode("utf-8").split("\n")]
    assert [(row["custom_id"], row["body"]["model"]) for row in rows] == [("a", "new-model"), ("b", "new-model")]


def test_replace_model_in_jsonl_with_embedded_newlines():
    """Test that replace_model_in_jsonl works correctly with embedded newlines in content"""
    # Create a JSONL with embedded newlines in the message content
    jsonl_data = {
        "custom_id": "test123",
        "body": {
            "model": "old-model",
            "messages": [{"role": "user", "content": "This is a message\nwith multiple\nlines"}],
        },
    }

    jsonl_bytes = json.dumps(jsonl_data).encode("utf-8")
    new_model = "new-model"

    result = replace_model_in_jsonl(jsonl_bytes, new_model)

    assert isinstance(result, InMemoryFile)

    # Read and parse the result
    result_content = result.read().decode("utf-8")
    result_json = json.loads(result_content)

    # Verify the model was replaced
    assert result_json["body"]["model"] == "new-model"
    # Verify the content with newlines is preserved
    assert result_json["body"]["messages"][0]["content"] == "This is a message\nwith multiple\nlines"
    assert result_json["custom_id"] == "test123"


def test_is_batch_retrieve_call_type_matches_only_batch_retrieves():
    from litellm.litellm_core_utils.batch_utils import is_batch_retrieve_call_type
    from litellm.types.utils import CallTypes

    assert is_batch_retrieve_call_type(CallTypes.aretrieve_batch.value) is True
    assert is_batch_retrieve_call_type(CallTypes.retrieve_batch.value) is True

    for call_type in CallTypes:
        if call_type in (CallTypes.aretrieve_batch, CallTypes.retrieve_batch):
            continue
        assert is_batch_retrieve_call_type(call_type.value) is False

    assert is_batch_retrieve_call_type(None) is False


@pytest.fixture(autouse=True)
def _vcr_outcome_gate(request, vcr):
    install_live_call_probe(request, vcr)
    yield
    record_vcr_outcome(request, vcr)


@pytest.fixture(scope="function", autouse=True)
def setup_and_teardown():
    """
    This fixture reloads litellm before every function. To speed up testing by removing callbacks being chained.
    """
    from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

    asyncio.run(GLOBAL_LOGGING_WORKER.clear_queue())
    importlib.reload(litellm)
    try:
        if hasattr(litellm, "proxy") and hasattr(litellm.proxy, "proxy_server"):
            importlib.reload(litellm.proxy.proxy_server)
    except Exception:
        pass
    loop = asyncio.get_event_loop_policy().new_event_loop()
    asyncio.set_event_loop(loop)
    yield
    loop.close()
    asyncio.set_event_loop(None)


def _batch(status: str, output_file_id: str | None = None, counts: BatchRequestCounts | None = None):
    from openai.types import Batch

    return Batch(
        id="job-budget", completion_window="24h", created_at=1, endpoint="/v1/chat/completions",
        input_file_id="file-budget", object="batch", status=status, output_file_id=output_file_id,
        request_counts=counts,
    )


def _completed_batch():
    return _batch("completed", output_file_id="file-out")


def _completed_video():
    from litellm.types.videos.main import VideoObject

    return VideoObject(id="job-budget", object="video", status="completed")


@pytest.mark.parametrize(
    ("call_type", "polled", "cost", "kind"),
    (
        ("aretrieve_batch", _completed_batch, 0.25, "batch"),
        ("retrieve_batch", _completed_batch, 0.25, "batch"),
        ("avideo_status", _completed_video, 0.25, "video"),
        ("video_status", _completed_video, 0.25, "video"),
        ("acompletion", _completed_batch, 0.25, None),
        ("avideo_status", _completed_batch, 0.25, None),
        ("aretrieve_batch", _completed_video, 0.25, None),
        ("avideo_generation", _completed_video, 0.25, None),
        ("aretrieve_batch", _completed_batch, 0, None),
        ("avideo_status", _completed_video, 0, None),
        ("avideo_status", _completed_video, -0.25, None),
    ),
)
def test_job_charge_identity_requires_a_cost_bearing_poll(call_type: str, polled, cost: float, kind) -> None:
    from litellm.litellm_core_utils.batch_utils import ChargeOnceJob, job_to_charge_once

    expected: Final = None if kind is None else ChargeOnceJob(kind=kind, job_id="job-budget")
    kwargs: Final = {
        "call_type": call_type,
        POLLED_VIDEO_STATUS_KWARG: "completed",
        "standard_logging_object": {"response_cost": cost},
    }
    assert job_to_charge_once(kwargs, polled(), cost) == expected
    assert job_to_charge_once(kwargs, {"id": "job-budget"}, cost) is None


@pytest.mark.parametrize(
    ("call_type", "polled", "guardrail_only_poll", "kind", "repeat_cost"),
    (
        ("aretrieve_batch", _completed_batch, lambda: _batch("in_progress"), "batch", 0.0),
        ("avideo_status", _completed_video, _completed_video, "video", 0.01),
    ),
)
def test_only_the_jobs_own_cost_is_charged_once_since_a_guardrail_charges_every_poll(
    call_type: str, polled, guardrail_only_poll, kind, repeat_cost
) -> None:
    from litellm.litellm_core_utils.batch_utils import ChargeOnceJob, job_to_charge_once

    kwargs: Final = {
        "call_type": call_type,
        POLLED_VIDEO_STATUS_KWARG: "completed",
        "standard_logging_object": {"guardrail_information": [{"guardrail_name": "priced", "guardrail_cost": 0.01}]},
    }
    assert job_to_charge_once(kwargs, guardrail_only_poll(), 0.01) is None
    assert job_to_charge_once(kwargs, polled(), 0.43) == ChargeOnceJob(
        kind=kind, job_id="job-budget", repeat_poll_cost=repeat_cost
    )


@pytest.mark.parametrize("status", ("completed", "failed", "cancelled", "expired"))
def test_a_finished_batch_that_cost_nothing_itself_is_still_charged_once_for_its_retrieves_guardrail(
    status: str,
) -> None:
    from litellm.litellm_core_utils.batch_utils import ChargeOnceJob, job_to_charge_once

    kwargs: Final = {
        "call_type": "aretrieve_batch",
        "standard_logging_object": {"guardrail_information": [{"guardrail_name": "priced", "guardrail_cost": 0.01}]},
    }
    finished_batch: Final = _batch(status, output_file_id="file-out")
    assert job_to_charge_once(kwargs, finished_batch, 0.01) == ChargeOnceJob(kind="batch", job_id="job-budget")


def test_a_completed_batch_still_waiting_on_its_output_does_not_take_the_charge_its_real_cost_needs() -> None:
    """Its first retrieve can report completed before the output file exists, and settling on that guardrail-only
    retrieve would leave the batch's real cost nothing to charge once the output arrives."""
    from litellm.litellm_core_utils.batch_utils import job_to_charge_once

    kwargs: Final = {
        "call_type": "aretrieve_batch",
        "standard_logging_object": {"guardrail_information": [{"guardrail_name": "priced", "guardrail_cost": 0.01}]},
    }
    assert job_to_charge_once(kwargs, _batch("completed"), 0.01) is None
    assert defers_batch_cost(kwargs, _batch("completed"))


def test_a_video_is_charged_once_it_completes_under_the_id_the_client_polled() -> None:
    """A provider may price a running video too, and a custom provider's own id can repeat across deployments."""
    from litellm.litellm_core_utils.batch_utils import ChargeOnceJob, job_to_charge_once
    from litellm.types.videos.main import VideoObject

    def poll(status: str, polled_id: str) -> ChargeOnceJob | None:
        return job_to_charge_once(
            {"call_type": "avideo_status", POLLED_VIDEO_STATUS_KWARG: status, "litellm_params": {"video_id": polled_id}},
            VideoObject(id="job-1", object="video", status=status),
            0.42,
        )

    assert poll("in_progress", "video_deployment_a") is None
    assert poll("completed", "video_deployment_a") == ChargeOnceJob(kind="video", job_id="video_deployment_a")
    assert poll("completed", "video_deployment_b") == ChargeOnceJob(kind="video", job_id="video_deployment_b")


@pytest.mark.parametrize("status", ["in_progress", "completed"])
def test_a_video_poll_is_judged_by_its_logged_status_when_redaction_replaced_the_video(status: str) -> None:
    from litellm.litellm_core_utils.batch_utils import ChargeOnceJob, job_to_charge_once

    kwargs: Final = {
        "call_type": "avideo_status",
        POLLED_VIDEO_STATUS_KWARG: status,
        "litellm_params": {"video_id": "video_deployment_a"},
    }
    expected: Final = ChargeOnceJob(kind="video", job_id="video_deployment_a") if status == "completed" else None

    assert job_to_charge_once(kwargs, {"text": "redacted-by-litellm"}, 0.42) == expected


@pytest.mark.parametrize(
    ("cost_is_final", "deferred", "expected"),
    [(False, True, None), (True, False, ChargeOnceJob(kind="batch", job_id="batch_abc"))],
)
def test_a_batch_retrieve_is_judged_by_its_logged_batch_when_redaction_replaced_it(
    cost_is_final: bool, deferred: bool, expected: ChargeOnceJob | None
) -> None:
    from litellm.litellm_core_utils.batch_utils import job_to_charge_once

    kwargs: Final = {
        "call_type": "aretrieve_batch",
        POLLED_BATCH_ID_KWARG: "batch_abc",
        POLLED_BATCH_COST_IS_FINAL_KWARG: cost_is_final,
        "standard_logging_object": {"guardrail_information": [{"guardrail_name": "priced", "guardrail_cost": 0.01}]},
    }
    redacted: Final = {"text": "redacted-by-litellm"}

    assert defers_batch_cost(kwargs, redacted) is deferred
    assert job_to_charge_once(kwargs, redacted, 0.01) == expected


def test_a_video_polled_with_or_without_its_base64_padding_is_charged_as_one_job() -> None:
    from litellm.litellm_core_utils.batch_utils import job_to_charge_once

    padded: Final = encode_video_id_with_provider("job-1", "custom_video", "deployment-a")
    assert padded.endswith("=")

    def charged_job(polled_id: str):
        return job_to_charge_once(
            {"call_type": "avideo_status", POLLED_VIDEO_STATUS_KWARG: "completed", "litellm_params": {"video_id": polled_id}},
            None,
            0.42,
        )

    assert charged_job(padded.rstrip("=")) == charged_job(padded)
    assert charged_job(padded) is not None


def test_job_charge_markers_keep_counters_jobs_and_kinds_independent() -> None:
    from litellm.caching.in_memory_cache import InMemoryCache
    from litellm.litellm_core_utils.batch_utils import ChargeOnceJob, job_charged_once_marker_key

    first_batch: Final = ChargeOnceJob(kind="batch", job_id="first-job")
    cache: Final = InMemoryCache()
    cache.set_cache(job_charged_once_marker_key("provider-spend", first_batch), 1)
    assert cache.get_cache(job_charged_once_marker_key("provider-spend", first_batch)) == 1
    assert cache.get_cache(job_charged_once_marker_key("deployment-spend", first_batch)) is None
    assert cache.get_cache(job_charged_once_marker_key("provider-spend", ChargeOnceJob("batch", "second-job"))) is None
    assert cache.get_cache(job_charged_once_marker_key("provider-spend", ChargeOnceJob("video", "first-job"))) is None


def test_batch_charge_marker_keeps_the_key_markers_already_written_use() -> None:
    """Markers written before the video kind existed must still block a re-charge after an upgrade."""
    from litellm.litellm_core_utils.batch_utils import ChargeOnceJob, job_charged_once_marker_key

    assert job_charged_once_marker_key("provider-spend", ChargeOnceJob("batch", "first-job")) == (
        "provider-spend:batch:first-job"
    )


def _retrieved_batch(
    status: str, output_file_id: str | None = None, counts: BatchRequestCounts | None = None
) -> LiteLLMBatch:
    return LiteLLMBatch(
        id="batch_abc",
        completion_window="24h",
        created_at=1,
        endpoint="/v1/chat/completions",
        input_file_id="file-in",
        object="batch",
        status="validating",
        output_file_id=output_file_id,
        request_counts=counts,
    ).model_copy(update={"status": status})


class TestBatchCostIsFinal:
    """Every retrieve of one batch writes the same spend row, so the first retrieve
    that prices it decides the row for good. A poll before the output exists must
    therefore not count as final: pricing it recorded $0 and pinned it (LIT-7048)."""

    @pytest.mark.parametrize("status", ["validating", "in_progress", "finalizing", "cancelling"])
    def test_in_flight_batch_is_not_final(self, status):
        assert batch_cost_is_final(_retrieved_batch(status)) is False

    @pytest.mark.parametrize("status", ["completed", "complete"])
    def test_completed_with_output_is_final(self, status):
        assert batch_cost_is_final(_retrieved_batch(status, output_file_id="file-out")) is True

    def test_completed_without_output_and_unknown_counts_is_not_final(self):
        assert batch_cost_is_final(_retrieved_batch("completed")) is False

    def test_completed_without_output_and_zero_counts_is_not_final(self):
        counts = BatchRequestCounts(total=0, completed=0, failed=0)
        assert batch_cost_is_final(_retrieved_batch("completed", counts=counts)) is False

    def test_completed_without_output_but_successful_lines_is_not_final(self):
        counts = BatchRequestCounts(total=2, completed=2, failed=0)
        assert batch_cost_is_final(_retrieved_batch("completed", counts=counts)) is False

    @pytest.mark.parametrize("status", ["completed", "complete"])
    def test_completed_without_output_and_every_line_failed_is_final(self, status):
        counts = BatchRequestCounts(total=2, completed=0, failed=2)
        assert batch_cost_is_final(_retrieved_batch(status, counts=counts)) is True

    @pytest.mark.parametrize("status", ["failed", "expired", "cancelled"])
    def test_other_terminal_statuses_are_final(self, status):
        assert batch_cost_is_final(_retrieved_batch(status)) is True
