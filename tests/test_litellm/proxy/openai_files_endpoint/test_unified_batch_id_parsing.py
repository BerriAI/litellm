"""Tests for unified batch ID parsing (native proxy + passthrough managed)."""

import base64

from litellm.proxy.openai_files_endpoints.common_utils import (
    get_batch_id_from_unified_batch_id,
    get_model_id_from_unified_batch_id,
    is_passthrough_unified_batch_id,
    parse_passthrough_unified_batch_id,
)
from litellm.proxy.pass_through_endpoints.managed_id_codec import encode


class TestPassthroughUnifiedBatchIdParsing:
    def test_parse_passthrough_plaintext(self):
        plaintext = (
            "litellm_proxy:passthrough;provider:azure;"
            "unified_id,abc-123;raw_id,batch_20260223-0518.234"
        )
        assert parse_passthrough_unified_batch_id(plaintext) == (
            "azure",
            "batch_20260223-0518.234",
        )
        assert is_passthrough_unified_batch_id(plaintext) is True

    def test_get_model_id_returns_provider_for_passthrough(self):
        plaintext = (
            "litellm_proxy:passthrough;provider:openai;"
            "unified_id,abc-123;raw_id,batch_xyz"
        )
        assert get_model_id_from_unified_batch_id(plaintext) == "openai"

    def test_get_batch_id_returns_raw_id_for_passthrough(self):
        plaintext = (
            "litellm_proxy:passthrough;provider:azure;"
            "unified_id,abc-123;raw_id,batch_abc"
        )
        assert get_batch_id_from_unified_batch_id(plaintext) == "batch_abc"

    def test_passthrough_base64_roundtrip(self):
        managed_id = encode("azure", "uuid-1", "batch_foundry_123")
        padded = managed_id + "=" * (-len(managed_id) % 4)
        plaintext = base64.urlsafe_b64decode(padded).decode()
        assert get_model_id_from_unified_batch_id(plaintext) == "azure"
        assert get_batch_id_from_unified_batch_id(plaintext) == "batch_foundry_123"

    def test_native_proxy_format_unchanged(self):
        native = (
            "litellm_proxy;model_id:deployment-123;"
            "llm_batch_id:batch_native_456;llm_output_file_id:file-out"
        )
        assert get_model_id_from_unified_batch_id(native) == "deployment-123"
        assert get_batch_id_from_unified_batch_id(native) == "batch_native_456"
        assert is_passthrough_unified_batch_id(native) is False

    def test_generic_response_id_format(self):
        generic = "litellm_proxy;model_id:dep-1;generic_response_id:batch_gen_1;"
        assert get_batch_id_from_unified_batch_id(generic) == "batch_gen_1"
        assert get_batch_id_from_unified_batch_id("not-a-batch-id") == ""
        assert get_model_id_from_unified_batch_id("not-a-batch-id") is None
