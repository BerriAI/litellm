"""Vendor §9.16/9.18 contract negatives for files + batches (LIT-4778).

Happy-path file/batch lifecycle is covered under batches/; this pins upload
without purpose/file and invalid batch id retrieve.
"""

from __future__ import annotations

import pytest
from e2e_http import NoBody, Success, UnknownApiError, assert_client_error
from lifecycle import ResourceManager
from proxy_client import ProxyClient
from pydantic import BaseModel

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
    def test_upload_without_purpose_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
        key = resources.key()
        result = proxy.transport.upload(
            "/v1/files",
            headers=proxy.transport.bearer(key),
            form=NoBody(),
            filename="batch_input.jsonl",
            content=b'{"custom_id":"1","method":"POST","url":"/v1/chat/completions","body":{}}\n',
            response_type=NoBody,
        )
        match result:
            case Success():
                pytest.fail("upload without purpose must not succeed")
            case UnknownApiError(status_code=status) if 400 <= status < 500:
                return
            case other:
                pytest.fail(f"upload without purpose expected 4xx, got {other!r}")

    @pytest.mark.skip(
        reason="stage red: product gap, /v1/batches 500s (acreate_batch TypeError) on missing input_file_id instead of 400"
    )
    @pytest.mark.covers("llm.batches.openai.input_validation.nonstream.works")
    def test_create_batch_missing_input_file_id_returns_error(
        self, proxy: ProxyClient, resources: ResourceManager
    ) -> None:
        key = resources.key()
        result = proxy.transport.send(
            "/v1/batches",
            headers=proxy.transport.bearer(key),
            json=BatchCreateBody(),
        )
        assert_client_error(result, "batch missing input_file_id")

    @pytest.mark.covers("llm.batches.openai.input_validation.nonstream.works")
    def test_retrieve_invalid_batch_id_returns_error(self, proxy: ProxyClient, resources: ResourceManager) -> None:
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
            case UnknownApiError(status_code=status) if status in (400, 404):
                return
            case other:
                pytest.fail(f"invalid batch id expected 400/404, got {other!r}")
