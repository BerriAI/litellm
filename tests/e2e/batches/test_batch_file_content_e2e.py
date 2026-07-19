"""Live e2e: GET /v1/files/{id}/content returns the uploaded batch JSONL."""

from __future__ import annotations

import pytest

from batch_client import BatchClient
from e2e_config import unique_marker
from e2e_http import FileUploadForm, unwrap
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from test_batches_e2e import quietly, render_jsonl

pytestmark = pytest.mark.e2e

OPENAI_BATCH_MODEL = "gpt-4o-mini"


class TestBatchFileContent:
    @pytest.mark.covers(
        "llm.files.openai.content.nonstream.works",
        exercised_on=["files"],
    )
    def test_file_content_matches_upload(
        self, client: BatchClient, resources: ResourceManager
    ) -> None:
        proxy_name = f"e2e-file-content-{unique_marker()}"
        model_id = client.create_model(
            proxy_name,
            LiteLLMParamsBody(
                model=f"openai/{OPENAI_BATCH_MODEL}",
                api_key="os.environ/OPENAI_API_KEY",
            ),
        )
        resources.defer(lambda: client.delete_model(model_id))
        key = resources.key()

        payload = render_jsonl(OPENAI_BATCH_MODEL)
        file = unwrap(
            client.upload_file(
                content=payload,
                form=FileUploadForm(purpose="batch", target_model_names=proxy_name),
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))
        assert file.id

        downloaded = client.proxy.transport.download(
            f"/v1/files/{file.id}/content",
            headers=client.proxy.transport.bearer(key),
        )
        assert downloaded.status_code == 200, (
            f"file content must be 200, got {downloaded.status_code}: {downloaded.body[:300]}"
        )
        expected = payload.decode().rstrip("\n")
        got = downloaded.body.rstrip("\n")
        assert got == expected, (
            "downloaded file content must match the uploaded JSONL bytes"
        )
