"""Managed batch-file upload guard for the LIT-3382 batch-processing failures.

The customer-facing symptom of LIT-3382 (and the OOM fixed in #31036) was that
uploading a batch JSONL failed outright, blocking batch workflows. This guard
drives the managed-files upload that works locally (the gemini AI Studio target),
end to end: it uploads a real multi-line OpenAI-format batch JSONL through the
proxy, decodes the unified file id the proxy returns, then reads the file back
from the provider's Files API and asserts it is ACTIVE and non-empty. A broken
upload or transform either 500s, returns an id with no backing provider file, or
persists an empty file - each fails an assertion here.

The faithful *memory*/OOM regression (the streaming transform #31036 actually
fixed) lives on the vertex_ai handler, which needs GCS + billing + memory
headroom; that guard is test_batch_upload_memory_e2e, enabled on EKS.

Skips when no proxy answers (shared conftest) or when GEMINI_API_KEY is absent
(the provider read-back needs it).
"""

import pytest

from batch_client import (
    GEMINI_FILES_PREFIX,
    BatchFilesClient,
    batch_jsonl,
    gemini_api_key,
    parse_unified_file_id,
)
from lifecycle import ResourceManager

pytestmark = pytest.mark.e2e

TARGET_MODEL = "gemini-2.5-flash"


def test_managed_batch_upload_persists_active_file_at_provider(
    client: BatchFilesClient, resources: ResourceManager, scoped_key: str
) -> None:
    api_key = gemini_api_key()
    if api_key is None:
        pytest.skip("GEMINI_API_KEY not set; cannot read the provider file back")

    content = batch_jsonl(TARGET_MODEL, lines=5)
    uploaded = client.upload_batch_file(scoped_key, content, target_model=TARGET_MODEL)
    resources.defer(lambda: client.delete_file(uploaded.id))

    assert uploaded.status == "uploaded"
    assert uploaded.purpose == "batch"
    assert uploaded.bytes is not None and uploaded.bytes > 0

    parsed = parse_unified_file_id(uploaded.id)
    assert TARGET_MODEL in parsed.target_models
    assert parsed.provider_file_uri.startswith(GEMINI_FILES_PREFIX)

    provider = client.provider_file(parsed.provider_file_uri, api_key=api_key)
    assert provider.state == "ACTIVE"
    assert provider.mimeType == "application/jsonl"
    assert provider.size_bytes > 0
