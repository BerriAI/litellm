import base64
from typing import Final
from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException

from litellm.proxy.guardrails.guardrail_hooks.azure import (
    AzureContentSafetyImageModerationGuardrail,
    guardrail_class_registry,
    guardrail_initializer_registry,
)
from litellm.proxy.guardrails.guardrail_registry import InMemoryGuardrailHandler
from litellm.types.guardrails import SupportedGuardrailIntegrations

IMAGE_BASE64: Final = base64.b64encode(b"small-image-bytes").decode()
IMAGE_DATA_URL: Final = f"data:image/png;base64,{IMAGE_BASE64}"
REMOTE_IMAGE_URL: Final = "https://example.com/cat.png"
AZURE_API_BASE: Final = "https://example.cognitiveservices.azure.com"


def _image_response(severity: int, category: str = "Violence") -> Mock:
    response = Mock()
    response.json.return_value = {
        "categoriesAnalysis": [
            {"category": "Hate", "severity": severity if category == "Hate" else 0},
            {"category": "Sexual", "severity": severity if category == "Sexual" else 0},
            {"category": "SelfHarm", "severity": severity if category == "SelfHarm" else 0},
            {"category": "Violence", "severity": severity if category == "Violence" else 0},
        ]
    }
    return response


def _image_guardrail(**kwargs) -> AzureContentSafetyImageModerationGuardrail:
    return AzureContentSafetyImageModerationGuardrail(
        guardrail_name="azure_image_moderation",
        api_key="azure_image_moderation_api_key",
        api_base=AZURE_API_BASE,
        **kwargs,
    )


@pytest.mark.asyncio
async def test_image_moderation_posts_base64_content():
    guardrail = _image_guardrail()

    with patch.object(guardrail.async_handler, "post", return_value=_image_response(0)) as mock_post:
        result = await guardrail.apply_guardrail(
            inputs={"images": [IMAGE_DATA_URL]},
            request_data={},
            input_type="request",
        )

    assert result == {"images": [IMAGE_DATA_URL]}
    assert mock_post.call_count == 1
    assert mock_post.call_args.kwargs["url"] == (f"{AZURE_API_BASE}/contentsafety/image:analyze?api-version=2024-09-01")
    body = mock_post.call_args.kwargs["json"]
    assert body["image"] == {"content": IMAGE_BASE64}
    assert body["categories"] == ["Hate", "SelfHarm", "Sexual", "Violence"]
    assert body["outputType"] == "FourSeverityLevels"


@pytest.mark.asyncio
async def test_image_moderation_rejects_remote_image_urls():
    guardrail = _image_guardrail()

    with patch.object(guardrail.async_handler, "post") as mock_post:
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"images": [REMOTE_IMAGE_URL]},
                request_data={},
                input_type="request",
            )

    mock_post.assert_not_called()
    assert exc_info.value.status_code == 400
    assert "remote image URLs are not supported" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_image_moderation_raises_on_violation():
    guardrail = _image_guardrail()

    with patch.object(guardrail.async_handler, "post", return_value=_image_response(4)):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"images": [IMAGE_DATA_URL]},
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 400
    assert "Violence crossed severity 2" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_image_moderation_honours_per_category_threshold():
    guardrail = _image_guardrail(severity_threshold_by_category={"Hate": 4})

    with patch.object(guardrail.async_handler, "post", return_value=_image_response(4, category="Violence")):
        await guardrail.apply_guardrail(
            inputs={"images": [IMAGE_DATA_URL]},
            request_data={},
            input_type="request",
        )

    with patch.object(guardrail.async_handler, "post", return_value=_image_response(4, category="Hate")):
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"images": [IMAGE_DATA_URL]},
                request_data={},
                input_type="request",
            )

    assert exc_info.value.status_code == 400
    assert "Hate crossed severity 4" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_image_moderation_skips_empty_images():
    guardrail = _image_guardrail()

    with patch.object(guardrail.async_handler, "post") as mock_post:
        result = await guardrail.apply_guardrail(inputs={"images": ["", ""]}, request_data={}, input_type="request")

    mock_post.assert_not_called()
    assert result == {"images": ["", ""]}


@pytest.mark.asyncio
async def test_image_moderation_rejects_plain_paths():
    guardrail = _image_guardrail()

    with patch.object(guardrail.async_handler, "post") as mock_post:
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"images": ["file.png"]},
                request_data={},
                input_type="request",
            )

    mock_post.assert_not_called()
    assert exc_info.value.status_code == 400
    assert "image must be a base64 data URL" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_image_moderation_rejects_data_url_without_base64():
    guardrail = _image_guardrail()

    with patch.object(guardrail.async_handler, "post") as mock_post:
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"images": ["data:image/png,not-base64"]},
                request_data={},
                input_type="request",
            )

    mock_post.assert_not_called()
    assert exc_info.value.status_code == 400
    assert "must be a base64 data URL" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_image_moderation_rejects_invalid_base64_payload():
    guardrail = _image_guardrail()

    with patch.object(guardrail.async_handler, "post") as mock_post:
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(
                inputs={"images": ["data:image/png;base64,not-base64!"]},
                request_data={},
                input_type="request",
            )

    mock_post.assert_not_called()
    assert exc_info.value.status_code == 400
    assert "not valid base64" in exc_info.value.detail["error"]


@pytest.mark.asyncio
async def test_image_moderation_rejects_oversized_images():
    guardrail = _image_guardrail()
    oversized = "data:image/png;base64," + base64.b64encode(b"x" * (4 * 1024 * 1024 + 1)).decode()

    with patch.object(guardrail.async_handler, "post") as mock_post:
        with pytest.raises(HTTPException) as exc_info:
            await guardrail.apply_guardrail(inputs={"images": [oversized]}, request_data={}, input_type="request")

    mock_post.assert_not_called()
    assert exc_info.value.status_code == 413
    assert "the limit is 4194304 bytes" in exc_info.value.detail["error"]


def test_image_moderation_registers_initializer_and_class():
    integration = SupportedGuardrailIntegrations.AZURE_IMAGE_MODERATIONS.value
    assert integration in guardrail_initializer_registry
    assert guardrail_class_registry[integration] is AzureContentSafetyImageModerationGuardrail


def test_image_moderation_event_hooks():
    assert AzureContentSafetyImageModerationGuardrail.get_supported_event_hooks() == [
        "pre_call",
        "post_call",
    ]


def test_image_moderation_uses_unified_lifecycle_dispatch():
    """Native lifecycle hooks are inherited no-ops, so pre and post call must go through apply_guardrail."""
    assert AzureContentSafetyImageModerationGuardrail.use_native_lifecycle_hooks is False


def test_image_moderation_loads_from_config():
    handler = InMemoryGuardrailHandler()
    registered = handler.initialize_guardrail(
        guardrail={
            "guardrail_name": "azure-image-moderation",
            "litellm_params": {
                "guardrail": "azure/image_moderations",
                "mode": "pre_call",
                "api_key": "azure_image_moderation_api_key",
                "api_base": AZURE_API_BASE,
            },
        }
    )

    assert registered is not None
    guardrail = handler.guardrail_id_to_custom_guardrail[registered["guardrail_id"]]
    assert isinstance(guardrail, AzureContentSafetyImageModerationGuardrail)
    assert guardrail.categories == ["Hate", "SelfHarm", "Sexual", "Violence"]
