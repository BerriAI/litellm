"""Live e2e: hosted_vllm file upload + batch create (OpenAI-compatible path).

hosted_vllm is in OPENAI_COMPATIBLE_BATCH_AND_FILES_PROVIDERS, so /v1/files and
/v1/batches route through the OpenAI handler against the deployment's api_base.
Requires HOSTED_VLLM_API_BASE (and optional HOSTED_VLLM_API_KEY) pointing at a
real vLLM (or OpenAI-compatible) server that implements the files/batches APIs.
"""

from __future__ import annotations

import os

import pytest

from batch_client import BatchClient, BatchCreateBody, BatchObject
from capabilities import batch_model_name
from e2e_config import require_env
from e2e_http import FileUploadForm, require_successful_call, unwrap
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from test_batches_e2e import (
    CREATED_BATCH_STATUSES,
    assert_batch_object,
    assert_file_object,
    quietly,
    render_jsonl,
)

pytestmark = pytest.mark.e2e


def _vllm_params(api_base: str, api_key: str | None, model_id: str) -> LiteLLMParamsBody:
    return LiteLLMParamsBody(
        model=f"hosted_vllm/{model_id}",
        api_base=api_base,
        api_key=api_key,
    )


class TestHostedVllmBatch:
    @pytest.mark.covers(
        "llm.batches.hosted_vllm.basic.nonstream.works",
        "llm.files.hosted_vllm.upload.nonstream.works",
        exercised_on=["batches", "files"],
    )
    def test_unified_file_and_batch_create(
        self, client: BatchClient, resources: ResourceManager
    ) -> None:
        (api_base,) = require_env("HOSTED_VLLM_API_BASE")
        api_key = (os.environ.get("HOSTED_VLLM_API_KEY") or "").strip() or None
        model_id = (os.environ.get("HOSTED_VLLM_MODEL") or "meta-llama/Llama-3.2-3B-Instruct").strip()
        proxy_name = batch_model_name("hosted-vllm-batch")

        model_row_id = client.create_model(
            proxy_name, _vllm_params(api_base, api_key, model_id)
        )
        resources.defer(lambda: client.delete_model(model_row_id))
        key = resources.key()

        file = unwrap(
            client.upload_file(
                content=render_jsonl(model_id),
                form=FileUploadForm(purpose="batch", target_model_names=proxy_name),
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))
        assert_file_object(file, provider="hosted_vllm")

        created = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)
        require_successful_call(created)
        batch = BatchObject.model_validate_json(created.body)
        resources.defer(quietly(lambda: client.cancel_batch(batch.id, key=key)))

        assert batch.id, f"hosted_vllm create returned no batch id: {created.body[:200]}"
        assert batch.status in CREATED_BATCH_STATUSES, (
            f"hosted_vllm batch has non-transitional status {batch.status!r}"
        )
        assert_batch_object(batch)
