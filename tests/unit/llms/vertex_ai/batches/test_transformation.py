"""
Unit tests for ``VertexAIBatchTransformation``
(litellm/llms/vertex_ai/batches/transformation.py).

This module is pure transformation logic: it maps OpenAI-shaped batch requests
into Vertex AI ``VertexAIBatchPredictionJob`` payloads, and maps Vertex AI batch
responses back into ``LiteLLMBatch`` / OpenAI list shapes. Unlike anthropic /
bedrock, this class does NOT subclass ``BaseBatchesConfig`` - it's a standalone
set of classmethods with a Vertex-specific shape, so these tests are fully
standalone and assert exact values rather than "ran without error".

There are no real I/O seams here; ``uuid.uuid4`` is the only nondeterministic
dependency and is patched where the displayName is asserted.
"""

from collections.abc import Mapping
from typing import Final
from unittest.mock import patch

import pytest

from litellm.llms.vertex_ai.batches.transformation import (  # noqa: E402
    VertexAIBatchTransformation,
)
from litellm.llms.vertex_ai.common_utils import (  # noqa: E402
    VertexAIError,
    _convert_vertex_datetime_to_openai_datetime,
)
from litellm.types.utils import LiteLLMBatch  # noqa: E402

T = VertexAIBatchTransformation

INPUT_FILE = (
    "gs://litellm-testing-bucket/litellm-vertex-files/publishers/google/"
    "models/gemini-1.5-flash-001/e9412502-2c91-42a6-8e61-f5c294cc0fc8"
)

ENDPOINT_ID = "7768560373388541952"
ENDPOINT_INPUT_FILE = (
    f"gs://litellm-testing-bucket/litellm-vertex-files/endpoints/{ENDPOINT_ID}/e9412502-2c91-42a6-8e61-f5c294cc0fc8"
)


def test_transform_openai_request_builds_full_vertex_job():
    with patch(
        "litellm.llms.vertex_ai.batches.transformation.uuid.uuid4",
        return_value="fixed-uuid",
    ):
        job = T.transform_openai_batch_request_to_vertex_ai_batch_request({"input_file_id": INPUT_FILE})

    assert job["displayName"] == "litellm-vertex-batch-fixed-uuid"
    assert job["model"] == "publishers/google/models/gemini-1.5-flash-001"

    assert job["inputConfig"]["instancesFormat"] == "jsonl"
    assert job["inputConfig"]["gcsSource"]["uris"] == [INPUT_FILE]

    assert job["outputConfig"]["predictionsFormat"] == "jsonl"
    # gcs uri prefix == file path with the filename stripped
    assert (
        job["outputConfig"]["gcsDestination"]["outputUriPrefix"]
        == "gs://litellm-testing-bucket/litellm-vertex-files/publishers/google/"
        "models/gemini-1.5-flash-001"
    )


def test_transform_openai_request_missing_input_file_id_raises():
    with pytest.raises(ValueError, match="input_file_id is required"):
        T.transform_openai_batch_request_to_vertex_ai_batch_request({})


def test_transform_openai_request_fine_tuned_endpoint_builds_endpoint_resource():
    """A fine-tuned Gemini file id (endpoints/<numeric id>) must target the endpoint resource,
    not a nonexistent publisher model (LIT-6899)."""
    job = T.transform_openai_batch_request_to_vertex_ai_batch_request(
        {"input_file_id": ENDPOINT_INPUT_FILE},
        vertex_project="my-project",
        vertex_location="us-central1",
    )
    assert job["model"] == f"projects/my-project/locations/us-central1/endpoints/{ENDPOINT_ID}"


def test_transform_openai_request_fine_tuned_endpoint_defaults_location():
    job = T.transform_openai_batch_request_to_vertex_ai_batch_request(
        {"input_file_id": ENDPOINT_INPUT_FILE},
        vertex_project="my-project",
    )
    assert job["model"] == f"projects/my-project/locations/us-central1/endpoints/{ENDPOINT_ID}"


def test_transform_openai_request_fine_tuned_endpoint_without_project_raises_400():
    with pytest.raises(VertexAIError) as exc_info:
        T.transform_openai_batch_request_to_vertex_ai_batch_request({"input_file_id": ENDPOINT_INPUT_FILE})
    assert exc_info.value.status_code == 400
    assert "vertex_project" in str(exc_info.value)


def test_transform_openai_request_publisher_model_ignores_project_and_location():
    job = T.transform_openai_batch_request_to_vertex_ai_batch_request(
        {"input_file_id": INPUT_FILE},
        vertex_project="my-project",
        vertex_location="europe-west4",
    )
    assert job["model"] == "publishers/google/models/gemini-1.5-flash-001"


@pytest.mark.parametrize(
    "input_file_id",
    [
        "gs://bucket/no-model-here.jsonl",
        "gs://bucket/publishers/google/gemini-1.5-flash-001/file-uuid",
        "gs://bucket/publishers/google/models",
        "gs://bucket/publishers/google/models//file-uuid",
    ],
)
def test_transform_openai_request_unparseable_model_raises_400(input_file_id: str):
    """An input_file_id with no parseable model path is a client error, not an IndexError -> 500."""
    with pytest.raises(VertexAIError) as exc_info:
        T.transform_openai_batch_request_to_vertex_ai_batch_request({"input_file_id": input_file_id})

    assert exc_info.value.status_code == 400
    assert input_file_id in str(exc_info.value)


# =========================================================================== #
# transform_vertex_ai_batch_response_to_openai_batch_response
# =========================================================================== #


def test_transform_vertex_response_full_mapping():
    response = {
        "name": "projects/510528649030/locations/us-central1/batchPredictionJobs/3814889423749775360",
        "state": "JOB_STATE_SUCCEEDED",
        "createTime": "2024-12-04T21:53:12.120184Z",
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": ["gs://bucket/in.jsonl"]},
        },
        "outputInfo": {"gcsOutputDirectory": "gs://bucket/out"},
    }
    batch = T.transform_vertex_ai_batch_response_to_openai_batch_response(response)

    assert isinstance(batch, LiteLLMBatch)
    assert batch.id == "3814889423749775360"
    assert batch.completion_window == "24h"
    # created_at is parsed via the shared helper (uses local tz); assert the
    # transform forwards createTime through that helper rather than a hardcoded
    # epoch that would be tz-dependent
    assert batch.created_at == _convert_vertex_datetime_to_openai_datetime("2024-12-04T21:53:12.120184Z")
    assert batch.endpoint == ""
    assert batch.object == "batch"
    assert batch.input_file_id == "gs://bucket/in.jsonl"
    assert batch.status == "completed"
    assert batch.error_file_id is None
    assert batch.output_file_id == "gs://bucket/out/predictions.jsonl"


def test_transform_vertex_response_error_file_id_always_none():
    batch = T.transform_vertex_ai_batch_response_to_openai_batch_response(
        {
            "name": "x/y/123",
            "state": "JOB_STATE_FAILED",
            "createTime": "2024-12-04T21:53:12.120184Z",
        }
    )
    assert batch.error_file_id is None


# =========================================================================== #
# _get_batch_job_status_from_vertex_ai_batch_response  (test EVERY entry)
# =========================================================================== #


@pytest.mark.parametrize(
    "vertex_state,expected",
    [
        ("JOB_STATE_UNSPECIFIED", "failed"),
        ("JOB_STATE_QUEUED", "validating"),
        ("JOB_STATE_PENDING", "validating"),
        ("JOB_STATE_RUNNING", "in_progress"),
        ("JOB_STATE_SUCCEEDED", "completed"),
        ("JOB_STATE_FAILED", "failed"),
        ("JOB_STATE_CANCELLING", "cancelling"),
        ("JOB_STATE_CANCELLED", "cancelled"),
        ("JOB_STATE_PAUSED", "in_progress"),
        ("JOB_STATE_EXPIRED", "expired"),
        ("JOB_STATE_UPDATING", "in_progress"),
        ("JOB_STATE_PARTIALLY_SUCCEEDED", "completed"),
    ],
)
def test_status_mapping_every_entry(vertex_state, expected):
    assert T._get_batch_job_status_from_vertex_ai_batch_response({"state": vertex_state}) == expected


def test_status_mapping_defaults_to_unspecified_when_missing():
    # No "state" key -> defaults to JOB_STATE_UNSPECIFIED -> "failed"
    assert T._get_batch_job_status_from_vertex_ai_batch_response({}) == "failed"


def test_status_mapping_unknown_state_raises_keyerror():
    with pytest.raises(KeyError):
        T._get_batch_job_status_from_vertex_ai_batch_response({"state": "NOPE"})


# =========================================================================== #
# _get_batch_id_from_vertex_ai_batch_response
# =========================================================================== #


def test_get_batch_id_splits_path():
    assert (
        T._get_batch_id_from_vertex_ai_batch_response({"name": "projects/p/locations/l/batchPredictionJobs/999"})
        == "999"
    )


def test_get_batch_id_no_slash_returns_name():
    assert T._get_batch_id_from_vertex_ai_batch_response({"name": "abc"}) == "abc"


def test_get_batch_id_empty_name_returns_empty():
    assert T._get_batch_id_from_vertex_ai_batch_response({"name": ""}) == ""
    assert T._get_batch_id_from_vertex_ai_batch_response({}) == ""


# =========================================================================== #
# _get_input_file_id_from_vertex_ai_batch_response
# =========================================================================== #


def test_get_input_file_id_happy_path():
    assert (
        T._get_input_file_id_from_vertex_ai_batch_response(
            {"inputConfig": {"gcsSource": {"uris": ["gs://b/a.jsonl", "gs://b/c.jsonl"]}}}
        )
        == "gs://b/a.jsonl"
    )


def test_get_input_file_id_missing_input_config():
    assert T._get_input_file_id_from_vertex_ai_batch_response({}) == ""


def test_get_input_file_id_missing_gcs_source():
    assert T._get_input_file_id_from_vertex_ai_batch_response({"inputConfig": {}}) == ""


def test_get_input_file_id_empty_uris():
    assert T._get_input_file_id_from_vertex_ai_batch_response({"inputConfig": {"gcsSource": {"uris": []}}}) == ""


# =========================================================================== #
# _get_output_file_id_from_vertex_ai_batch_response: None until Vertex reports outputInfo
# =========================================================================== #

SHARED_OUTPUT_PREFIX: Final = "gs://bucket/litellm-vertex-files/publishers/google/models/gemini-2.5-flash"
SUCCEEDED_OUTPUT_DIRECTORY: Final = f"{SHARED_OUTPUT_PREFIX}/prediction-model-2026-09-24T19:41:00.000000Z"


def _vertex_job(state: str) -> dict[str, object]:
    return {
        "name": "projects/510528649030/locations/us-central1/batchPredictionJobs/3814889423749775360",
        "state": state,
        "createTime": "2026-09-24T19:37:25.775603Z",
        "inputConfig": {
            "instancesFormat": "jsonl",
            "gcsSource": {"uris": [f"{SHARED_OUTPUT_PREFIX}/0586ba52-4f8b-4988-aa8d-3573550a4b0f"]},
        },
        "outputConfig": {
            "predictionsFormat": "jsonl",
            "gcsDestination": {"outputUriPrefix": SHARED_OUTPUT_PREFIX},
        },
    }


@pytest.mark.parametrize(
    "vertex_state,output_info_field,expected_status,expected_output_file_id",
    [
        ("JOB_STATE_PENDING", {}, "validating", None),
        ("JOB_STATE_RUNNING", {"outputInfo": {}}, "in_progress", None),
        ("JOB_STATE_CANCELLED", {"outputInfo": None}, "cancelled", None),
        (
            "JOB_STATE_SUCCEEDED",
            {"outputInfo": {"gcsOutputDirectory": SUCCEEDED_OUTPUT_DIRECTORY}},
            "completed",
            f"{SUCCEEDED_OUTPUT_DIRECTORY}/predictions.jsonl",
        ),
    ],
    ids=["create_or_pending", "running", "cancelled", "succeeded"],
)
def test_transform_vertex_response_output_file_id_is_none_until_output_info(
    vertex_state: str,
    output_info_field: Mapping[str, object],
    expected_status: str,
    expected_output_file_id: str | None,
) -> None:
    batch: Final = T.transform_vertex_ai_batch_response_to_openai_batch_response(
        {**_vertex_job(vertex_state), **output_info_field}
    )

    assert batch.status == expected_status
    assert batch.output_file_id == expected_output_file_id


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"outputConfig": {}},
        {"outputInfo": None},
        {"outputInfo": {"gcsOutputDirectory": ""}},
        {"outputInfo": {"gcsOutputDirectory": None}},
    ],
    ids=[
        "no_fields",
        "output_config_without_destination",
        "null_output_info",
        "empty_output_directory",
        "null_output_directory",
    ],
)
def test_get_output_file_id_is_none_without_output_directory(response: Mapping[str, object]) -> None:
    assert T._get_output_file_id_from_vertex_ai_batch_response(response) is None


def test_get_output_file_id_from_output_info():
    # outputInfo branch: rstrip trailing slash, append predictions.jsonl
    assert (
        T._get_output_file_id_from_vertex_ai_batch_response({"outputInfo": {"gcsOutputDirectory": "gs://bucket/out/"}})
        == "gs://bucket/out/predictions.jsonl"
    )


def test_get_output_file_id_output_info_no_trailing_slash():
    assert (
        T._get_output_file_id_from_vertex_ai_batch_response({"outputInfo": {"gcsOutputDirectory": "gs://bucket/out"}})
        == "gs://bucket/out/predictions.jsonl"
    )


def test_get_output_file_id_output_info_ignores_output_uri_prefix():
    resp = {
        "outputInfo": {"gcsOutputDirectory": "gs://from-info"},
        "outputConfig": {"gcsDestination": {"outputUriPrefix": "gs://from-config"}},
    }
    assert T._get_output_file_id_from_vertex_ai_batch_response(resp) == "gs://from-info/predictions.jsonl"


# =========================================================================== #
# _get_gcs_uri_prefix_from_file
# =========================================================================== #


def test_get_gcs_uri_prefix_root():
    assert (
        T._get_gcs_uri_prefix_from_file("gs://litellm-testing-bucket/vtx_batch.jsonl") == "gs://litellm-testing-bucket"
    )


def test_get_gcs_uri_prefix_nested():
    assert (
        T._get_gcs_uri_prefix_from_file("gs://litellm-testing-bucket/batches/vtx_batch.jsonl")
        == "gs://litellm-testing-bucket/batches"
    )


# =========================================================================== #
# _get_model_from_gcs_file
# =========================================================================== #


def test_get_model_from_gcs_file_plain():
    assert T._get_model_from_gcs_file(INPUT_FILE) == "publishers/google/models/gemini-1.5-flash-001"


def test_get_model_from_gcs_file_url_encoded():
    # %2F decodes to "/" via urllib.unquote before splitting
    encoded = "gs://bucket/publishers%2Fgoogle%2Fmodels%2Fgemini-1.5-flash-001%2Fuuid"
    assert T._get_model_from_gcs_file(encoded) == "publishers/google/models/gemini-1.5-flash-001"


def test_get_model_from_gcs_file_no_publishers_raises_400():
    with pytest.raises(VertexAIError) as exc_info:
        T._get_model_from_gcs_file("gs://bucket/no-model-here.jsonl")
    assert exc_info.value.status_code == 400


def test_get_model_from_gcs_file_fine_tuned_endpoint():
    """The whole endpoint id must survive parsing; the old 3-segment publishers/ parse dropped it."""
    assert T._get_model_from_gcs_file(ENDPOINT_INPUT_FILE) == f"endpoints/{ENDPOINT_ID}"


def test_get_model_from_gcs_file_publisher_path_wins_over_endpoints_prefix():
    """A bucket prefix containing endpoints/<digits> must not override the publisher model path
    LiteLLM appended after it."""
    uri = "gs://bucket/team-endpoints/999/litellm-vertex-files/publishers/google/models/gemini-1.5-flash-001/uuid"
    assert T._get_model_from_gcs_file(uri) == "publishers/google/models/gemini-1.5-flash-001"


def test_get_model_from_gcs_file_last_endpoints_segment_wins():
    """With no publisher path, the endpoint id closest to the file (last occurrence) is the one
    LiteLLM stored; an earlier prefix segment must not shadow it."""
    uri = f"gs://bucket/endpoints/999/litellm-vertex-files/endpoints/{ENDPOINT_ID}/uuid"
    assert T._get_model_from_gcs_file(uri) == f"endpoints/{ENDPOINT_ID}"


def test_get_model_from_gcs_file_non_numeric_endpoints_segment_raises_400():
    with pytest.raises(VertexAIError) as exc_info:
        T._get_model_from_gcs_file("gs://bucket/endpoints/not-a-number/file-uuid")
    assert exc_info.value.status_code == 400


def test_get_bare_model_name_from_gcs_file_fine_tuned_endpoint():
    assert T.get_bare_model_name_from_gcs_file(ENDPOINT_INPUT_FILE) == ENDPOINT_ID


# =========================================================================== #
# is_unmanaged_gcs_batch_input_file_id
# =========================================================================== #


@pytest.mark.parametrize(
    "input_file_id, expected",
    [
        (INPUT_FILE, True),
        (None, False),
        ("file-abc123", False),
        ("gs://bucket/no-model-here.jsonl", False),
        ("gs://bucket/publishers/google/gemini-1.5-flash-001/file-uuid", False),
        (ENDPOINT_INPUT_FILE, True),
        ("gs://bucket/endpoints/not-a-number/file-uuid", False),
    ],
)
def test_is_unmanaged_gcs_batch_input_file_id(input_file_id, expected):
    assert T.is_unmanaged_gcs_batch_input_file_id(input_file_id) is expected


# =========================================================================== #
# transform_vertex_ai_batch_list_response_to_openai_list_response
# =========================================================================== #


def _job(batch_id: str) -> dict:
    return {
        "name": f"projects/p/locations/l/batchPredictionJobs/{batch_id}",
        "state": "JOB_STATE_SUCCEEDED",
        "createTime": "2024-12-04T21:53:12.120184Z",
    }


def test_list_response_multiple_jobs():
    response = {
        "batchPredictionJobs": [_job("111"), _job("222"), _job("333")],
        "nextPageToken": "tok-abc",
    }
    out = T.transform_vertex_ai_batch_list_response_to_openai_list_response(response)

    assert out["object"] == "list"
    assert [b.id for b in out["data"]] == ["111", "222", "333"]
    assert out["first_id"] == "111"
    assert out["last_id"] == "333"
    assert out["has_more"] is True
    assert out["next_page_token"] == "tok-abc"


def test_list_response_no_next_page_token():
    response = {"batchPredictionJobs": [_job("111")]}
    out = T.transform_vertex_ai_batch_list_response_to_openai_list_response(response)
    assert out["has_more"] is False
    assert out["next_page_token"] is None
    assert out["first_id"] == "111"
    assert out["last_id"] == "111"


def test_list_response_empty():
    out = T.transform_vertex_ai_batch_list_response_to_openai_list_response({})
    assert out["data"] == []
    assert out["first_id"] is None
    assert out["last_id"] is None
    assert out["has_more"] is False


def test_list_response_none_jobs_treated_as_empty():
    out = T.transform_vertex_ai_batch_list_response_to_openai_list_response({"batchPredictionJobs": None})
    assert out["data"] == []
    assert out["first_id"] is None


PASSTHROUGH_INPUT_FILE = (
    "gs://litellm-testing-bucket/litellm-vertex-files/passthrough/publishers/google/models/gemini-2.5-flash/uuid-1"
)


def test_get_model_from_passthrough_gcs_file():
    assert T._get_model_from_gcs_file(PASSTHROUGH_INPUT_FILE) == "publishers/google/models/gemini-2.5-flash"


def test_get_gcs_uri_prefix_keeps_passthrough_segment_so_output_lands_beside_input():
    assert (
        T._get_gcs_uri_prefix_from_file(PASSTHROUGH_INPUT_FILE)
        == "gs://litellm-testing-bucket/litellm-vertex-files/passthrough/publishers/google/models/gemini-2.5-flash"
    )
