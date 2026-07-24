"""Vendor §9.16/9.18 contract negatives for files + batches (LIT-4778).

Happy-path file/batch lifecycle is covered under batches/; this pins upload
without purpose/file and invalid batch id retrieve.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from e2e_config import unique_marker
from e2e_http import NoBody, Success, UnknownApiError
from lifecycle import ResourceManager
from models import LiteLLMParamsBody
from proxy_client import ProxyClient
from vendor_contract import assert_error_or_server_known

pytestmark = pytest.mark.e2e


class BatchCreateBody(BaseModel):
    input_file_id: str | None = None
    endpoint: str = "/v1/chat/completions"
    completion_window: str = "24h"


class BatchObject(BaseModel):
    id: str
    status: str | None = None


class TestFilesBatchesContract:
    @pytest.mark.covers("llm.files.openai.input_validation.nonstream.works")
    def test_upload_without_purpose_returns_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-files-contract-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key = resources.key()

        class EmptyForm(BaseModel):
            pass

        result = proxy.transport.upload(
            "/v1/files",
            headers=proxy.transport.bearer(key),
            form=EmptyForm(),
            filename="batch_input.jsonl",
            content=b'{"custom_id":"1","method":"POST","url":"/v1/chat/completions","body":{}}\n',
            response_type=NoBody,
        )
        match result:
            case Success():
                pytest.fail("upload without purpose must not succeed")
            case UnknownApiError(status_code=status):
                assert status in range(400, 600), f"unexpected {status}"
            case _:
                return

    @pytest.mark.covers("llm.batches.openai.input_validation.nonstream.works")
    def test_create_batch_missing_input_file_id_returns_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-batch-contract-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key = resources.key()
        result = proxy.transport.send(
            "/v1/batches",
            headers=proxy.transport.bearer(key),
            json=BatchCreateBody(),
        )
        assert_error_or_server_known(result, "batch missing input_file_id")

    @pytest.mark.covers("llm.batches.openai.input_validation.nonstream.works")
    def test_retrieve_invalid_batch_id_returns_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        model = f"e2e-batch-contract-{unique_marker()}"
        model_id = proxy.create_model(
            model,
            LiteLLMParamsBody(model="openai/gpt-4o-mini", api_key="os.environ/OPENAI_API_KEY"),
        )
        resources.defer(lambda: proxy.delete_model(model_id))
        key = resources.key()
        result = proxy.transport.get(
            "/v1/batches/invalid-batch-id",
            headers=proxy.transport.bearer(key),
            params=NoBody(),
            response_type=BatchObject,
        )
        match result:
            case Success():
                pytest.fail("invalid batch id must not succeed")
            case UnknownApiError(status_code=status):
                assert status in (400, 404, 500), f"unexpected {status}"
            case _:
                return
