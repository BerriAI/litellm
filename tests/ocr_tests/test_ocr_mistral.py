"""
Test OCR functionality with Mistral API.
"""
import os
import sys
import pytest
import litellm



# Use a simple test image or PDF URL
TEST_IMAGE_PATH = "test_image_edit.png"
TEST_PDF_URL = "https://arxiv.org/pdf/2201.04234"


class TestMistralOCR:
    """
    Test class for Mistral OCR functionality.
    """

    def get_base_ocr_call_args(self) -> dict:
        """Return the base OCR call args for Mistral"""
        return {
            "model": "mistral/mistral-ocr-latest",
            "api_key": os.getenv("MISTRAL_API_KEY"),
        }

    @pytest.mark.parametrize("sync_mode", [True, False])
    @pytest.mark.asyncio
    async def test_basic_mistral_ocr_with_url(self, sync_mode):
        """
        Test basic OCR with a public URL using Mistral native format.
        """
        litellm.set_verbose = True
        base_ocr_call_args = self.get_base_ocr_call_args()

        try:
            if sync_mode:
                response = litellm.ocr(
                    document={
                        "type": "document_url",
                        "document_url": TEST_PDF_URL
                    },
                    **base_ocr_call_args,
                )
            else:
                response = await litellm.aocr(
                    document={
                        "type": "document_url",
                        "document_url": TEST_PDF_URL
                    },
                    **base_ocr_call_args,
                )

            print(f"\n{'='*80}")
            print(f"Sync Mode: {sync_mode}")
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
            pytest.fail(f"OCR call failed: {str(e)}")

    def test_mistral_ocr_response_structure(self):
        """
        Test that the OCR response has the correct Mistral format structure.
        """
        litellm.set_verbose = True
        base_ocr_call_args = self.get_base_ocr_call_args()

        response = litellm.ocr(
            document={
                "type": "document_url",
                "document_url": TEST_PDF_URL
            },
            **base_ocr_call_args,
        )

        # Validate response structure (Mistral format)
        assert hasattr(response, "pages"), "Response should have 'pages' attribute"
        assert hasattr(response, "model"), "Response should have 'model' attribute"
        assert hasattr(response, "object"), "Response should have 'object' attribute"
        assert hasattr(response, "usage_info"), "Response should have 'usage_info' attribute"
        
        assert isinstance(response.pages, list), "pages should be a list"
        assert len(response.pages) > 0, "Should have at least one page"
        assert response.object == "ocr", "object should be 'ocr'"
        
        # Validate first page structure
        first_page = response.pages[0]
        assert hasattr(first_page, "index"), "Page should have 'index' attribute"
        assert hasattr(first_page, "markdown"), "Page should have 'markdown' attribute"
        assert isinstance(first_page.markdown, str), "markdown should be a string"
        
        print(f"\nResponse structure validated:")
        print(f"  - object: {response.object}")
        print(f"  - model: {response.model}")
        print(f"  - pages: {len(response.pages)}")
        if response.usage_info:
            print(f"  - pages_processed: {response.usage_info.pages_processed}")
            print(f"  - doc_size_bytes: {response.usage_info.doc_size_bytes}")

