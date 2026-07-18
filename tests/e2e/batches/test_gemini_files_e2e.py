"""Live e2e: Gemini Files API upload through the proxy.

gemini is a first-class FileCreateProvider. The test registers a gemini
deployment, uploads a tiny batch-purpose JSONL with target_model_names routing,
and asserts a FileObject comes back. Batch create for pure gemini (non-Vertex)
is out of scope here; Vertex covers the Gemini batch job path in the main
lifecycle matrix.
"""

from __future__ import annotations

import pytest

from batch_client import BatchClient
from capabilities import batch_model_name
from e2e_http import FileUploadForm, unwrap
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from test_batches_e2e import assert_file_object, quietly, render_jsonl

pytestmark = pytest.mark.e2e

RAW_MODEL = "gemini-2.5-flash"


class TestGeminiFiles:
    @pytest.mark.covers(
        "llm.files.gemini.upload.nonstream.works",
        exercised_on=["files"],
    )
    def test_gemini_file_upload(
        self, client: BatchClient, resources: ResourceManager
    ) -> None:
        model_name = batch_model_name("gemini-files")
        model_id = client.create_model(
            model_name,
            LiteLLMParamsBody(
                model=f"gemini/{RAW_MODEL}",
                api_key="os.environ/GEMINI_API_KEY",
            ),
        )
        resources.defer(lambda: client.delete_model(model_id))
        key = resources.key()

        file = unwrap(
            client.upload_file(
                content=render_jsonl(RAW_MODEL),
                form=FileUploadForm(purpose="batch", target_model_names=model_name),
                key=key,
            )
        )
        resources.defer(quietly(lambda: client.delete_file(file.id, key=key)))
        assert_file_object(file, provider="gemini")
        assert file.id, "gemini file upload returned no id"
