"""
Test OCR functionality with Mistral API.
"""
import os
import sys
import pytest
import litellm
from litellm import Router
from base_ocr_unit_tests import BaseOCRTest, TEST_PDF_URL


class TestMistralOCR(BaseOCRTest):
    """
    Test class for Mistral OCR functionality.
    """

    def get_base_ocr_call_args(self) -> dict:
        """Return the base OCR call args for Mistral"""
        return {
            "model": "mistral/mistral-ocr-latest",
            "api_key": os.getenv("MISTRAL_API_KEY"),
        }

@pytest.mark.asyncio
async def test_router_aocr_with_mistral():
    """
    Test OCR with Router using Mistral OCR deployment.
    """
    litellm.set_verbose = True

    # Create router with Mistral OCR deployment
    router = Router(
        model_list=[
            {
                "model_name": "mistral-ocr",
                "litellm_params": {
                    "model": "mistral/mistral-ocr-latest",
                    "api_key": os.getenv("MISTRAL_API_KEY"),
                },
            }
        ]
    )

    try:
        # Call OCR through router
        response = await router.aocr(
            model="mistral-ocr",
            document={
                "type": "document_url",
                "document_url": TEST_PDF_URL
            },
        )

        print(f"\n{'='*80}")
        print("Router OCR Test")
        print(f"Response type: {type(response)}")
        print(f"Response object: {response.object if hasattr(response, 'object') else 'N/A'}")
        
        # Check if response has expected Mistral OCR format
        assert hasattr(response, "pages"), "Response should have 'pages' attribute"
        assert hasattr(response, "model"), "Response should have 'model' attribute"
        assert hasattr(response, "object"), "Response should have 'object' attribute"
        assert response.object == "ocr", f"Expected object='ocr', got '{response.object}'"
        
        # Validate pages structure
        assert isinstance(response.pages, list), "pages should be a list"
        assert len(response.pages) > 0, "Should have at least one page"
        
        # Check first page structure
        first_page = response.pages[0]
        assert hasattr(first_page, "index"), "Page should have 'index' attribute"
        assert hasattr(first_page, "markdown"), "Page should have 'markdown' attribute"
        
        # Extract text from all pages for validation
        total_text = "\n\n".join(page.markdown for page in response.pages if page.markdown)
        print(f"Total pages: {len(response.pages)}")
        print(f"Total extracted text length: {len(total_text)} characters")
        print(f"First 200 chars: {total_text[:200]}")
        print(f"Model: {response.model}")
        if response.usage_info:
            print(f"Pages processed: {response.usage_info.pages_processed}")
        print(f"{'='*80}\n")
        
        assert len(total_text) > 0, "Should extract some text from the document"

    except Exception as e:
        pytest.fail(f"Router OCR call failed: {str(e)}")

