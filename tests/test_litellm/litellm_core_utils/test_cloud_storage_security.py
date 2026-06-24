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
