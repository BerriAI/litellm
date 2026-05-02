"""
VERIA-51: polling URLs returned by upstream APIs (Azure DALL-E,
Azure Document Intelligence, Black Forest Labs) used to be followed
without origin validation. The handlers attached the operator's API
key to the polling request, so an attacker who could influence the
upstream response (or a compromised upstream) could redirect the proxy
to send credentials anywhere.

These tests assert each handler now rejects polling URLs that don't
share an origin with the original request URL.
"""

from unittest.mock import MagicMock, patch

import httpx
import pytest


# Azure DALL-E sync + async paths route through ``assert_same_origin``
# the same way as the cases below. The helper itself is unit-tested in
# ``tests/test_litellm/litellm_core_utils/test_url_utils.py``; the
# tests here exercise the wiring at sites with simpler signatures.


# ── Azure Document Intelligence polling ───────────────────────────────────────


def test_azure_di_sync_rejects_cross_origin_polling():
    from litellm.llms.azure_ai.ocr.document_intelligence.transformation import (
        AzureDocumentIntelligenceOCRConfig,
    )

    config = AzureDocumentIntelligenceOCRConfig()

    raw_response = MagicMock()
    raw_response.status_code = 202
    raw_response.headers = {
        "Operation-Location": "https://attacker.example.com/results/xyz",
    }
    raw_response.request = MagicMock()
    raw_response.request.url = (
        "https://eastus.cognitiveservices.azure.com/documentintelligence/.../analyze"
    )
    raw_response.request.headers = {"Ocp-Apim-Subscription-Key": "leak-me"}

    with pytest.raises(ValueError, match="rejected polling URL"):
        config.transform_ocr_response(
            model="azure-doc-intel",
            raw_response=raw_response,
            logging_obj=MagicMock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
            response={},
        )


# ── Black Forest Labs polling ─────────────────────────────────────────────────


def test_bfl_image_generation_sync_rejects_cross_origin_polling():
    from litellm.llms.black_forest_labs.image_generation.handler import (
        BlackForestLabsImageGeneration,
    )

    handler = BlackForestLabsImageGeneration()

    initial_response = MagicMock()
    initial_response.status_code = 200
    initial_response.json = MagicMock(
        return_value={"polling_url": "https://attacker.example.com/get_result"}
    )
    initial_response.request = MagicMock()
    initial_response.request.url = "https://api.bfl.ai/v1/flux-pro"

    sync_client = MagicMock()
    sync_client.get = MagicMock()

    with pytest.raises(Exception, match="Rejected polling URL"):
        handler._poll_for_result_sync(
            initial_response=initial_response,
            headers={"x-key": "secret"},
            sync_client=sync_client,
        )

    sync_client.get.assert_not_called()


@pytest.mark.asyncio
async def test_bfl_image_generation_async_rejects_cross_origin_polling():
    from litellm.llms.black_forest_labs.image_generation.handler import (
        BlackForestLabsImageGeneration,
    )

    handler = BlackForestLabsImageGeneration()

    initial_response = MagicMock()
    initial_response.status_code = 200
    initial_response.json = MagicMock(
        return_value={"polling_url": "https://attacker.example.com/get_result"}
    )
    initial_response.request = MagicMock()
    initial_response.request.url = "https://api.bfl.ai/v1/flux-pro"

    async_client = MagicMock()
    async_client.get = MagicMock()

    with pytest.raises(Exception, match="Rejected polling URL"):
        await handler._poll_for_result_async(
            initial_response=initial_response,
            headers={"x-key": "secret"},
            async_client=async_client,
        )

    async_client.get.assert_not_called()


def test_bfl_image_edit_sync_rejects_cross_origin_polling():
    from litellm.llms.black_forest_labs.image_edit.handler import (
        BlackForestLabsImageEdit,
    )

    handler = BlackForestLabsImageEdit()

    initial_response = MagicMock()
    initial_response.status_code = 200
    initial_response.json = MagicMock(
        return_value={"polling_url": "https://attacker.example.com/get_result"}
    )
    initial_response.request = MagicMock()
    initial_response.request.url = "https://api.bfl.ai/v1/flux-pro/edit"

    sync_client = MagicMock()
    sync_client.get = MagicMock()

    with pytest.raises(Exception, match="Rejected polling URL"):
        handler._poll_for_result_sync(
            initial_response=initial_response,
            headers={"x-key": "secret"},
            sync_client=sync_client,
        )

    sync_client.get.assert_not_called()


def test_bfl_image_generation_same_origin_polling_passes():
    """Sanity check: when the polling URL shares origin with the original
    request, the origin check passes and polling proceeds."""
    from litellm.llms.black_forest_labs.image_generation.handler import (
        BlackForestLabsImageGeneration,
    )

    handler = BlackForestLabsImageGeneration()

    initial_response = MagicMock()
    initial_response.status_code = 200
    initial_response.json = MagicMock(
        return_value={"polling_url": "https://api.bfl.ai/v1/get_result?id=abc"}
    )
    initial_response.request = MagicMock()
    initial_response.request.url = "https://api.bfl.ai/v1/flux-pro"

    sync_client = MagicMock()
    poll_response = MagicMock()
    poll_response.status_code = 200
    poll_response.json = MagicMock(return_value={"status": "Ready"})
    sync_client.get = MagicMock(return_value=poll_response)

    result = handler._poll_for_result_sync(
        initial_response=initial_response,
        headers={"x-key": "secret"},
        sync_client=sync_client,
    )

    sync_client.get.assert_called_once()
    assert result is poll_response
