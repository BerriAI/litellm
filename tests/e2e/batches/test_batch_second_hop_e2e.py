from __future__ import annotations

import pytest
from batch_cleanup import cleanup_batch, cleanup_file
from batch_client import BatchClient, BatchCreateBody, BatchObject
from batch_helpers import CREATED_BATCH_STATUSES, assert_batch_object, render_jsonl, retrieve_batch
from capabilities import OPENAI_BATCH_BACKEND, OPENAI_BATCH_MODEL, batch_model_name, is_managed_id
from e2e_config import PROXY_BASE_URL
from e2e_http import FileUploadForm, require_successful_call, unwrap
from e2e_metadata import Domain, Mode, Route, Subject, meta
from e2e_metadata import Provider as MetaProvider
from lifecycle import ResourceManager
from models import LiteLLMParamsBody

pytestmark = pytest.mark.e2e


class TestBatchSecondHop:
    """Two-proxy batch routing: a litellm_proxy deployment chained to the gateway
    itself (LIT-5347, PR #36240).

    The hop deployment's litellm_params point litellm_proxy/<inner model> at this
    gateway's own base URL with a freshly minted virtual key, so the unified
    upload and batch create traverse gateway -> gateway -> OpenAI. The regression
    this pins: target_model_names must be rewritten to the inner deployment on
    the second hop and the nested managed ids must round-trip retrieve.
    """

    @pytest.mark.covers(
        "llm.batches.openai.second_hop.nonstream.works",
        exercised_on=["batches", "files"],
    )
    @meta(
        Subject(
            domain=Domain.LLM_TRANSLATION,
            route=Route.BATCHES,
            providers=(MetaProvider.LITELLM_PROXY, MetaProvider.OPENAI),
            models=(OPENAI_BATCH_BACKEND,),
            mode=Mode.BATCH,
        )
    )
    def test_unified_create_and_retrieve_via_chained_gateway(
        self, client: BatchClient, resources: ResourceManager, batch_deployments: None
    ) -> None:
        key = resources.key()
        hop_name = batch_model_name("openai-batch-hop")
        model_id = client.create_model(
            hop_name,
            LiteLLMParamsBody(
                model=f"litellm_proxy/{OPENAI_BATCH_MODEL}",
                api_base=PROXY_BASE_URL,
                api_key=key,
            ),
        )
        resources.defer(lambda: client.delete_model(model_id))

        file = unwrap(
            client.upload_file(
                content=render_jsonl(OPENAI_BATCH_BACKEND),
                form=FileUploadForm(purpose="batch", target_model_names=hop_name),
                key=key,
            )
        )
        resources.defer(lambda: cleanup_file(client, file.id, key=key))
        assert is_managed_id(file.id), (
            f"second-hop unified upload must return a managed file id, got {file.id!r}"
        )

        created = client.create_batch(body=BatchCreateBody(input_file_id=file.id), key=key)
        require_successful_call(created)
        batch = BatchObject.model_validate_json(created.body)
        resources.defer(lambda: cleanup_batch(client, batch.id, key=key))

        assert is_managed_id(batch.id), (
            f"second-hop create must return a managed batch id, got {batch.id!r}"
        )
        assert batch.status in CREATED_BATCH_STATUSES, (
            f"second-hop batch has non-transitional status {batch.status!r}"
        )
        assert_batch_object(batch)

        fetched = retrieve_batch(client, batch.id, key=key, provider=None)
        assert fetched.id == batch.id
        assert fetched.status, "second-hop retrieve returned no status"
