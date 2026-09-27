import json
from typing import Final

import pytest

from litellm.llms.xai.batches.transformation import (
    XAIBatch,
    XAIBatchesError,
    XAIBatchList,
    XAIBatchResult,
    XAIBatchResultsPage,
    get_xai_api_base,
    results_to_openai_jsonl,
    to_create_batch_body,
    to_litellm_batch,
    to_openai_batch_list,
    xai_batches_url,
)
from litellm.types.llms.openai import CreateBatchRequest

SEPT_23_2026_UTC: Final = 1790121600


def _xai_batch(**overrides: object) -> XAIBatch:
    return XAIBatch.model_validate(
        {
            "batch_id": "batch_9bdf",
            "name": "nightly",
            "create_time": "2026-09-23",
            "expire_time": "2026-10-23",
            "cancel_time": None,
            "cancel_by_xai_message": None,
            "state": {"num_requests": 2, "num_pending": 0, "num_success": 2, "num_error": 0, "num_cancelled": 0},
            "input_file_id": "file_07",
            **overrides,
        }
    )


def test_completed_batch_exposes_batch_id_as_output_file_and_maps_counts() -> None:
    batch: Final = to_litellm_batch(_xai_batch())

    assert batch.model_dump(exclude_none=True) == {
        "id": "batch_9bdf",
        "object": "batch",
        "endpoint": "/v1/chat/completions",
        "input_file_id": "file_07",
        "completion_window": "24h",
        "status": "completed",
        "created_at": SEPT_23_2026_UTC,
        "expires_at": SEPT_23_2026_UTC + 30 * 86400,
        "output_file_id": "batch_9bdf",
        "request_counts": {"total": 2, "completed": 2, "failed": 0},
        "metadata": {"name": "nightly"},
    }


def test_pending_requests_mean_in_progress_and_no_output_file() -> None:
    batch: Final = to_litellm_batch(
        _xai_batch(state={"num_requests": 3, "num_pending": 1, "num_success": 1, "num_error": 1, "num_cancelled": 0})
    )

    assert (batch.status, batch.output_file_id) == ("in_progress", None)
    assert batch.request_counts is not None
    assert batch.request_counts.model_dump() == {"total": 3, "completed": 1, "failed": 1}


def test_empty_batch_is_still_validating() -> None:
    assert to_litellm_batch(_xai_batch(state={})).status == "validating"


def test_batch_cancelled_by_xai_validation_is_failed_with_the_message() -> None:
    batch: Final = to_litellm_batch(
        _xai_batch(
            state={},
            cancel_time="2026-09-23T10:00:00Z",
            cancel_by_xai_message="JSONL file validation failed: Model grok-nope is not supported",
        )
    )

    assert batch.status == "failed"
    assert batch.failed_at == SEPT_23_2026_UTC + 10 * 3600
    assert batch.cancelled_at is None
    assert batch.errors is not None and batch.errors.data is not None
    assert [e.message for e in batch.errors.data] == ["JSONL file validation failed: Model grok-nope is not supported"]


def test_batch_cancelled_by_caller_is_cancelled() -> None:
    batch: Final = to_litellm_batch(_xai_batch(cancel_time="2026-09-23"))

    assert (batch.status, batch.cancelled_at, batch.errors) == ("cancelled", SEPT_23_2026_UTC, None)


@pytest.mark.parametrize(
    "endpoint",
    [
        "/v1/images/generations",
        "/v1/images/edits",
        "/v1/videos/generations",
        "/v1/videos/edits",
        "/v1/videos/extensions",
    ],
)
def test_create_body_accepts_image_and_video_endpoints(endpoint: str) -> None:
    body: Final = to_create_batch_body(
        CreateBatchRequest(completion_window="24h", endpoint=endpoint, input_file_id="file_07")
    )

    assert dict(body) == {"name": "litellm-batch", "input_file_id": "file_07"}


def test_create_body_uses_input_file_id_and_metadata_name() -> None:
    body: Final = to_create_batch_body(
        CreateBatchRequest(
            completion_window="24h", endpoint="/v1/chat/completions", input_file_id="file_07", metadata={"name": "n1"}
        )
    )

    assert dict(body) == {"name": "n1", "input_file_id": "file_07"}


def test_create_body_without_input_file_id_is_a_400() -> None:
    with pytest.raises(XAIBatchesError) as exc:
        to_create_batch_body(CreateBatchRequest(completion_window="24h", endpoint="/v1/chat/completions"))

    assert exc.value.status_code == 400


def test_results_render_as_openai_output_jsonl_with_errors_per_line() -> None:
    page: Final = XAIBatchResultsPage.model_validate(
        {
            "results": [
                {
                    "batch_request_id": "r1",
                    "batch_result": {
                        "response": {
                            "chat_get_completion": {"id": "c1", "object": "chat.completion", "choices": [], "usage": {}}
                        }
                    },
                },
                {"batch_request_id": "r2", "batch_result": {"error": {"code": 3, "message": "bad model"}}},
                {"batch_request_id": "r3", "batch_result": {}},
            ],
            "pagination_token": None,
        }
    )

    lines: Final = [json.loads(line) for line in results_to_openai_jsonl(page.results).decode().splitlines()]

    assert lines == [
        {
            "id": "batch_req_r1",
            "custom_id": "r1",
            "response": {
                "status_code": 200,
                "request_id": "c1",
                "body": {"id": "c1", "object": "chat.completion", "choices": [], "usage": {}},
            },
            "error": None,
        },
        {"id": "batch_req_r2", "custom_id": "r2", "response": None, "error": {"code": "3", "message": "bad model"}},
        {
            "id": "batch_req_r3",
            "custom_id": "r3",
            "response": None,
            "error": {"code": "request_failed", "message": "xAI returned no response for this request"},
        },
    ]


@pytest.mark.parametrize(
    ("response_key", "body"),
    [
        ("responses", {"id": "resp_1", "output": []}),
        ("image_generation", {"created": 1, "data": [{"url": "https://cdn.example/img.png"}]}),
        ("video_generation", {"id": "vid_1", "url": "https://cdn.example/clip.mp4"}),
    ],
)
def test_result_unwraps_the_single_response_key_into_the_openai_body(
    response_key: str, body: dict[str, object]
) -> None:
    result: Final = XAIBatchResult.model_validate(
        {"batch_request_id": "r", "batch_result": {"response": {response_key: body}}}
    )

    line: Final = json.loads(results_to_openai_jsonl((result,)).decode())
    assert line["response"]["body"] == body
    assert line["response"]["request_id"] == body.get("id")
    assert response_key not in line["response"]["body"]


def test_retrieve_and_list_report_chat_because_xai_has_no_batch_endpoint() -> None:
    retrieved: Final = to_litellm_batch(_xai_batch())
    listed: Final = to_openai_batch_list(XAIBatchList.model_validate({"batches": [_xai_batch().model_dump()]}))

    assert retrieved.endpoint == "/v1/chat/completions"
    assert [batch.endpoint for batch in listed.data] == ["/v1/chat/completions"]
    assert retrieved.metadata == {"name": "nightly"}


def test_list_page_maps_to_openai_list_with_cursor_flags() -> None:
    page: Final = XAIBatchList.model_validate(
        {"batches": [_xai_batch().model_dump(), _xai_batch(batch_id="batch_2").model_dump()], "pagination_token": "t"}
    )

    listed: Final = to_openai_batch_list(page)

    assert (listed.object, listed.first_id, listed.last_id, listed.has_more, listed.next_page_token) == (
        "list",
        "batch_9bdf",
        "batch_2",
        True,
        "t",
    )
    assert [b.id for b in listed.data] == ["batch_9bdf", "batch_2"]


@pytest.mark.parametrize(
    "api_base", ["https://api.x.ai", "https://api.x.ai/", "https://api.x.ai/v1", "https://api.x.ai/v1/"]
)
def test_api_base_never_doubles_the_v1_segment(api_base: str) -> None:
    assert get_xai_api_base(api_base) == "https://api.x.ai"
    assert xai_batches_url(api_base, "batch_1", ":cancel") == "https://api.x.ai/v1/batches/batch_1:cancel"
    assert xai_batches_url(api_base) == "https://api.x.ai/v1/batches"
