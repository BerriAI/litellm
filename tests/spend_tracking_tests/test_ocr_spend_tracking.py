"""
Unit tests for OCR spend tracking in get_logging_payload.

This test file verifies that OCR/AOCR calls correctly extract usage_info
and populate the spend logs payload with pages_processed instead of token counts.
"""
import pytest
from datetime import datetime, timezone
from unittest.mock import Mock
from pydantic import BaseModel
from typing import Optional

from litellm.proxy.spend_tracking.spend_tracking_utils import (
    get_logging_payload,
    _extract_usage_for_ocr_call,
)


class MockUsageInfo(BaseModel):
    """Mock Pydantic model for OCR usage_info"""
    pages_processed: int
    doc_size_bytes: Optional[int] = None


class MockOCRResponse(BaseModel):
    """Mock Pydantic model for OCR response"""
    id: str
    object: str
    model: str
    usage_info: MockUsageInfo


class TestExtractUsageForOCRCall:
    """Test the _extract_usage_for_ocr_call helper method"""

    def test_extract_usage_from_dict(self):
        """Test extracting usage from dict response"""
        response_obj_dict = {
            "usage_info": {
                "pages_processed": 5
            }
        }
        
        usage = _extract_usage_for_ocr_call(response_obj_dict, response_obj_dict)
        
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0
        assert usage["pages_processed"] == 5

    def test_extract_usage_from_pydantic_model(self):
        """Test extracting usage from Pydantic model response"""
        usage_info = MockUsageInfo(pages_processed=10, doc_size_bytes=1024)
        response_obj = MockOCRResponse(
            id="ocr-123",
            object="ocr",
            model="test-ocr-model",
            usage_info=usage_info
        )
        response_obj_dict = response_obj.model_dump()
        
        usage = _extract_usage_for_ocr_call(response_obj, response_obj_dict)
        
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 0
        assert usage["total_tokens"] == 0
        assert usage["pages_processed"] == 10

    def test_extract_usage_with_object_attributes(self):
        """Test extracting usage from object with __dict__"""
        class SimpleUsageInfo:
            def __init__(self, pages_processed):
                self.pages_processed = pages_processed
        
        class SimpleOCRResponse:
            def __init__(self):
                self.usage_info = SimpleUsageInfo(pages_processed=3)
        
        response_obj = SimpleOCRResponse()
        response_obj_dict = {}
        
        usage = _extract_usage_for_ocr_call(response_obj, response_obj_dict)
        
        assert usage.get("prompt_tokens") == 0
        assert usage.get("completion_tokens") == 0
        assert usage.get("total_tokens") == 0
        assert usage.get("pages_processed") == 3

    def test_extract_usage_missing_usage_info(self):
        """Test handling missing usage_info"""
        response_obj_dict = {}
        
        usage = _extract_usage_for_ocr_call(response_obj_dict, response_obj_dict)
        
        assert usage == {}

    def test_extract_usage_empty_usage_info(self):
        """Test handling empty usage_info"""
        response_obj_dict = {
            "usage_info": {}
        }
        
        usage = _extract_usage_for_ocr_call(response_obj_dict, response_obj_dict)
        
        assert usage.get("prompt_tokens") == 0
        assert usage.get("completion_tokens") == 0
        assert usage.get("total_tokens") == 0
        assert usage.get("pages_processed") == 0


class TestGetLoggingPayloadOCR:
    """Test get_logging_payload with OCR call types"""

    @pytest.fixture
    def mock_datetime(self):
        """Fixture for consistent timestamps"""
        return datetime.now(timezone.utc)

    @pytest.fixture
    def base_kwargs(self):
        """Fixture for base kwargs used in tests"""
        return {
            "model": "test-ocr-model",
            "call_type": "ocr",
            "litellm_params": {},
            "response_cost": 0.05,
        }

    def test_ocr_call_with_dict_response(self, mock_datetime, base_kwargs):
        """Test OCR call with dict response containing usage_info"""
        response_obj = {
            "id": "ocr-test-123",
            "object": "ocr",
            "model": "test-ocr-model",
            "usage_info": {
                "pages_processed": 7,
                "doc_size_bytes": 2048
            }
        }
        
        payload = get_logging_payload(
            kwargs=base_kwargs,
            response_obj=response_obj,
            start_time=mock_datetime,
            end_time=mock_datetime
        )
        
        assert payload["call_type"] == "ocr"
        assert payload["prompt_tokens"] == 0
        assert payload["completion_tokens"] == 0
        assert payload["total_tokens"] == 0
        assert payload["spend"] == 0.05
        
        # Verify pages_processed is in additional_usage_values
        import json
        metadata = json.loads(payload["metadata"])
        assert "additional_usage_values" in metadata
        assert metadata["additional_usage_values"]["pages_processed"] == 7

    def test_aocr_call_with_pydantic_response(self, mock_datetime, base_kwargs):
        """Test AOCR (async OCR) call with Pydantic model response"""
        base_kwargs["call_type"] = "aocr"
        
        usage_info = MockUsageInfo(pages_processed=12)
        response_obj = MockOCRResponse(
            id="aocr-test-456",
            object="ocr",
            model="test-ocr-model",
            usage_info=usage_info
        )
        
        payload = get_logging_payload(
            kwargs=base_kwargs,
            response_obj=response_obj,
            start_time=mock_datetime,
            end_time=mock_datetime
        )
        
        assert payload["call_type"] == "aocr"
        assert payload["prompt_tokens"] == 0
        assert payload["completion_tokens"] == 0
        assert payload["total_tokens"] == 0
        
        # Verify pages_processed is in additional_usage_values
        import json
        metadata = json.loads(payload["metadata"])
        assert "additional_usage_values" in metadata
        assert metadata["additional_usage_values"]["pages_processed"] == 12

    def test_ocr_call_missing_usage_info(self, mock_datetime, base_kwargs):
        """Test OCR call with missing usage_info returns empty usage"""
        response_obj = {
            "id": "ocr-test-789",
            "object": "ocr",
            "model": "test-ocr-model"
        }
        
        payload = get_logging_payload(
            kwargs=base_kwargs,
            response_obj=response_obj,
            start_time=mock_datetime,
            end_time=mock_datetime
        )
        
        assert payload["call_type"] == "ocr"
        assert payload["prompt_tokens"] == 0
        assert payload["completion_tokens"] == 0
        assert payload["total_tokens"] == 0

    def test_ocr_call_with_zero_pages(self, mock_datetime, base_kwargs):
        """Test OCR call with zero pages processed"""
        response_obj = {
            "id": "ocr-test-000",
            "object": "ocr",
            "model": "test-ocr-model",
            "usage_info": {
                "pages_processed": 0
            }
        }
        
        payload = get_logging_payload(
            kwargs=base_kwargs,
            response_obj=response_obj,
            start_time=mock_datetime,
            end_time=mock_datetime
        )
        
        assert payload["call_type"] == "ocr"
        assert payload["prompt_tokens"] == 0
        assert payload["completion_tokens"] == 0
        assert payload["total_tokens"] == 0
        
        # Verify pages_processed is 0
        import json
        metadata = json.loads(payload["metadata"])
        assert metadata["additional_usage_values"]["pages_processed"] == 0

    def test_non_ocr_call_uses_token_based_usage(self, mock_datetime):
        """Test that non-OCR calls still use token-based usage"""
        kwargs = {
            "model": "gpt-4",
            "call_type": "completion",
            "litellm_params": {},
            "response_cost": 0.02,
        }
        
        response_obj = {
            "id": "completion-test-123",
            "object": "chat.completion",
            "model": "gpt-4",
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 100,
                "total_tokens": 150
            }
        }
        
        payload = get_logging_payload(
            kwargs=kwargs,
            response_obj=response_obj,
            start_time=mock_datetime,
            end_time=mock_datetime
        )
        
        assert payload["call_type"] == "completion"
        assert payload["prompt_tokens"] == 50
        assert payload["completion_tokens"] == 100
        assert payload["total_tokens"] == 150

    def test_ocr_with_metadata(self, mock_datetime, base_kwargs):
        """Test OCR call with additional metadata"""
        base_kwargs["litellm_params"] = {
            "metadata": {
                "user_api_key_user_id": "test-user",
                "user_api_key_team_id": "test-team"
            }
        }
        
        response_obj = {
            "id": "ocr-metadata-test",
            "object": "ocr",
            "model": "test-ocr-model",
            "usage_info": {
                "pages_processed": 5,
                "doc_size_bytes": 1024
            }
        }
        
        payload = get_logging_payload(
            kwargs=base_kwargs,
            response_obj=response_obj,
            start_time=mock_datetime,
            end_time=mock_datetime
        )
        
        assert payload["call_type"] == "ocr"
        assert payload["user"] == "test-user"
        assert payload["prompt_tokens"] == 0
        assert payload["completion_tokens"] == 0
        
        # Verify pages_processed and doc_size_bytes are both in additional_usage_values
        import json
        metadata = json.loads(payload["metadata"])
        assert metadata["additional_usage_values"]["pages_processed"] == 5
        assert metadata["additional_usage_values"]["doc_size_bytes"] == 1024
