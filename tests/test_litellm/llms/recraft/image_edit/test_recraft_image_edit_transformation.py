import json
import os
import sys
from io import BufferedReader, BytesIO
from typing import Dict, List
from unittest.mock import MagicMock, mock_open, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.recraft.image_edit.transformation import RecraftImageEditConfig
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageObject, ImageResponse


class TestRecraftImageEditTransformation:
    """
    Unit tests for Recraft image edit transformation functionality.
    """
    
    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.config = RecraftImageEditConfig()
        self.model = "recraft-v3"
        self.logging_obj = MagicMock()
        self.prompt = "Add more trees to this landscape"

    def test_transform_image_edit_request(self):
        """
        Test that transform_image_edit_request correctly transforms request parameters 
        and separates files from data.
        """
        # Mock image data
        image_data = b"fake_image_data"
        image = BytesIO(image_data)
        
        image_edit_optional_params = {
            "n": 2,
            "response_format": "url",
            "strength": 0.5,
            "style": "photographic"
        }
        
        litellm_params = GenericLiteLLMParams()
        headers = {}
        
        data, files = self.config.transform_image_edit_request(
            model=self.model,
            prompt=self.prompt,
            image=image,
            image_edit_optional_request_params=image_edit_optional_params,
            litellm_params=litellm_params,
            headers=headers
        )
        
        # Check that data contains the expected parameters
        assert data["model"] == self.model
        assert data["prompt"] == self.prompt
        assert data["strength"] == 0.5
        assert data["n"] == 2
        assert data["response_format"] == "url"
        assert data["style"] == "photographic"
        
        # Check that image is not in data (should be in files)
        assert "image" not in data
        
        # Check that files contains the image
        assert len(files) == 1
        assert files[0][0] == "image"  # field name
        assert files[0][1][0] == "image.png"  # filename (default for non-BufferedReader)
        assert files[0][1][1] == image  # file object

    def test_get_image_files_for_request_single_image(self):
        """
        Test that _get_image_files_for_request correctly handles a single image.
        """
        image_data = b"fake_image_data"
        image = BytesIO(image_data)
        
        files = self.config._get_image_files_for_request(image=image)
        
        assert len(files) == 1
        assert files[0][0] == "image"
        assert files[0][1][0] == "image.png"  # Default filename for non-BufferedReader
        assert files[0][1][1] == image
        assert "image/png" in files[0][1][2]

    def test_get_image_files_for_request_list_with_single_image(self):
        """
        Test that _get_image_files_for_request correctly handles a list containing a single image
        (takes the first image for Recraft API).
        """
        image_data = b"fake_image_data"
        image = BytesIO(image_data)
        
        # Pass as list (OpenAI format)
        files = self.config._get_image_files_for_request(image=[image])
        
        assert len(files) == 1
        assert files[0][0] == "image"
        assert files[0][1][0] == "image.png"  # Default filename for non-BufferedReader
        assert files[0][1][1] == image

    def test_get_image_files_for_request_buffered_reader(self):
        """
        Test that _get_image_files_for_request correctly handles BufferedReader objects.
        """
        # Create a mock BufferedReader
        mock_file = MagicMock(spec=BufferedReader)
        mock_file.name = "buffered_image.jpg"
        
        files = self.config._get_image_files_for_request(image=mock_file)
        
        assert len(files) == 1
        assert files[0][0] == "image"
        assert files[0][1][0] == "buffered_image.jpg"
        assert files[0][1][1] == mock_file

    def test_get_image_files_for_request_no_image(self):
        """
        Test that _get_image_files_for_request returns empty list when no image is provided.
        """
        files = self.config._get_image_files_for_request(image=None)
        assert files == []

    def test_transform_image_edit_response_success(self):
        """
        Test that transform_image_edit_response correctly transforms a successful response.
        """
        # Mock response data
        response_data = {
            "data": [
                {"url": "https://example.com/edited_image1.jpg", "b64_json": None},
                {"url": None, "b64_json": "base64encodeddata"}
            ]
        }
        
        # Create mock response
        mock_response = MagicMock()
        mock_response.json.return_value = response_data
        
        result = self.config.transform_image_edit_response(
            model=self.model,
            raw_response=mock_response,
            logging_obj=self.logging_obj
        )
        
        assert isinstance(result, ImageResponse)
        assert len(result.data) == 2
        assert result.data[0].url == "https://example.com/edited_image1.jpg"
        assert result.data[0].b64_json is None
        assert result.data[1].url is None
        assert result.data[1].b64_json == "base64encodeddata"

    def test_transform_image_edit_response_json_error(self):
        """
        Test that transform_image_edit_response raises appropriate error when response JSON is invalid.
        """
        # Create mock response that raises JSON decode error
        mock_response = MagicMock()
        mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
        mock_response.status_code = 500
        mock_response.headers = {}
        
        with pytest.raises(Exception) as exc_info:
            self.config.transform_image_edit_response(
                model=self.model,
                raw_response=mock_response,
                logging_obj=self.logging_obj
            )
        
        assert "Error transforming image edit response" in str(exc_info.value) 