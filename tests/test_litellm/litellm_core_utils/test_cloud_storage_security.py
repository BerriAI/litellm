import pytest

from litellm.litellm_core_utils.cloud_storage_security import (
    is_managed_cloud_storage_uri,
)


def test_is_managed_cloud_storage_uri_detects_raw_object_uris():
    assert is_managed_cloud_storage_uri("s3://bucket/litellm-batch-outputs/x.jsonl.out")
    assert is_managed_cloud_storage_uri("gs://bucket/litellm-vertex-files/x")


def test_is_managed_cloud_storage_uri_ignores_provider_and_unified_ids():
    # Plain provider ids and base64 unified ids carry no storage scheme.
    assert not is_managed_cloud_storage_uri("file-abc123")
    assert not is_managed_cloud_storage_uri("bGl0ZWxsbV9wcm94eQ==")
    assert not is_managed_cloud_storage_uri("")


@pytest.mark.parametrize(
    "file_id",
    (
        "gs%3A%2F%2Fbucket%2Flitellm-vertex-files%2Fprediction-model%2Fpredictions.jsonl",
        "gs%253A%252F%252Fbucket%252Flitellm-vertex-files%252Fprediction-model%252Fpredictions.jsonl",
        "s3%3A%2F%2Fbucket%2Flitellm-batch-outputs%2Fx.jsonl.out",
        "s3%253A%252F%252Fbucket%252Flitellm-batch-outputs%252Fx.jsonl.out",
    ),
)
def test_is_managed_cloud_storage_uri_sees_through_percent_encoding(file_id: str):
    assert is_managed_cloud_storage_uri(file_id)


def test_is_managed_cloud_storage_uri_survives_an_id_encoded_thousands_of_times():
    nested_gs_id = "gs" + "%" + "25" * 5000 + "3A//bucket/x"
    nested_plain_id = "%" + "25" * 5000

    assert is_managed_cloud_storage_uri(nested_gs_id)
    assert not is_managed_cloud_storage_uri(nested_plain_id)
