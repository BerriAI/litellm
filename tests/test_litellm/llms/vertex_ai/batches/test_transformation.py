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

import os
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.vertex_ai.batches.transformation import (  # noqa: E402
    VertexAIBatchTransformation,
)
from litellm.llms.vertex_ai.common_utils import (  # noqa: E402
    _convert_vertex_datetime_to_openai_datetime,
)
from litellm.types.utils import LiteLLMBatch  # noqa: E402

T = VertexAIBatchTransformation

INPUT_FILE = (
    "gs://litellm-testing-bucket/litellm-vertex-files/publishers/google/"
    "models/gemini-1.5-flash-001/e9412502-2c91-42a6-8e61-f5c294cc0fc8"
)


# =========================================================================== #
# transform_openai_batch_request_to_vertex_ai_batch_request
# =========================================================================== #


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
# _get_output_file_id_from_vertex_ai_batch_response
# =========================================================================== #


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


def test_get_output_file_id_empty_output_info_falls_through_to_output_config():
    # gcsOutputDirectory missing -> "" -> the "/predictions.jsonl" guard skips
    # the outputInfo branch, falls through to outputConfig
    resp = {
        "outputInfo": {},
        "outputConfig": {"gcsDestination": {"outputUriPrefix": "gs://b/cfg"}},
    }
    assert T._get_output_file_id_from_vertex_ai_batch_response(resp) == "gs://b/cfg/predictions.jsonl"


def test_get_output_file_id_no_output_info_and_no_output_config():
    assert T._get_output_file_id_from_vertex_ai_batch_response({}) == ""


def test_get_output_file_id_output_config_missing_gcs_destination():
    # outputConfig present but no gcsDestination -> returns the running "" value
    assert T._get_output_file_id_from_vertex_ai_batch_response({"outputConfig": {}}) == ""


def test_get_output_file_id_output_config_already_has_suffix():
    # outputUriPrefix already ends in /predictions.jsonl -> returned as-is (no double append)
    resp = {"outputConfig": {"gcsDestination": {"outputUriPrefix": "gs://b/cfg/predictions.jsonl"}}}
    assert T._get_output_file_id_from_vertex_ai_batch_response(resp) == "gs://b/cfg/predictions.jsonl"


def test_get_output_file_id_output_config_strips_trailing_slash():
    resp = {"outputConfig": {"gcsDestination": {"outputUriPrefix": "gs://b/cfg/"}}}
    assert T._get_output_file_id_from_vertex_ai_batch_response(resp) == "gs://b/cfg/predictions.jsonl"


def test_get_output_file_id_output_info_takes_precedence_over_output_config():
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


def test_get_model_from_gcs_file_no_publishers_raises():
    with pytest.raises(IndexError):
        T._get_model_from_gcs_file("gs://bucket/no-model-here.jsonl")


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
