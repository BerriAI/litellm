from io import BufferedReader, BytesIO
from typing import Dict

import pytest

from litellm import image_edit
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.types.router import GenericLiteLLMParams


@pytest.fixture
def image_edit_config() -> OpenAIImageEditConfig:
    return OpenAIImageEditConfig()


def test_transform_image_edit_request_basic(image_edit_config: OpenAIImageEditConfig):
    """Test basic image edit request transformation with image and prompt"""
    model = "dall-e-2"
    prompt = "Make the background blue"
    image = b"fake_image_data"
    image_edit_optional_request_params = {}
    litellm_params = GenericLiteLLMParams()
    headers = {}

    data, files = image_edit_config.transform_image_edit_request(
        model=model,
        prompt=prompt,
        image=image,
        image_edit_optional_request_params=image_edit_optional_request_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    # Check that data contains model and prompt but not image
    assert data["model"] == model
    assert data["prompt"] == prompt
    assert "image" not in data
    assert "mask" not in data

    # Check that files contains the image
    assert len(files) == 1
    assert files[0][0] == "image[]"  # field name
    assert files[0][1][0] == "image.png"  # filename
    assert files[0][1][1] == image  # image data
    assert "image/png" in files[0][1][2]  # content type


def test_transform_image_edit_request_with_mask(image_edit_config: OpenAIImageEditConfig):
    """Test transformation with mask parameter"""
    model = "dall-e-2"
    prompt = "Make the background blue"
    image = b"fake_image_data"
    mask = b"fake_mask_data"
    image_edit_optional_request_params = {"mask": mask, "size": "1024x1024"}
    litellm_params = GenericLiteLLMParams()
    headers = {}

    data, files = image_edit_config.transform_image_edit_request(
        model=model,
        prompt=prompt,
        image=image,
        image_edit_optional_request_params=image_edit_optional_request_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    # Check that data contains model, prompt, and size but not image or mask
    assert data["model"] == model
    assert data["prompt"] == prompt
    assert data["size"] == "1024x1024"
    assert "image" not in data
    assert "mask" not in data

    # Check that files contains both image and mask
    assert len(files) == 2
    
    # Find image and mask in files
    image_file = next(f for f in files if f[0] == "image[]")
    mask_file = next(f for f in files if f[0] == "mask")
    
    assert image_file[1][0] == "image.png"
    assert image_file[1][1] == image
    assert "image/png" in image_file[1][2]
    
    assert mask_file[1][0] == "mask.png"
    assert mask_file[1][1] == mask
    assert "image/png" in mask_file[1][2]


def test_transform_image_edit_request_with_buffered_reader(image_edit_config: OpenAIImageEditConfig):
    """Test transformation with BufferedReader as image input"""
    import os
    import tempfile
    
    model = "dall-e-2"
    prompt = "Make the background blue"
    
    # Create a real file to get a proper BufferedReader
    image_data = b"fake_image_data"
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_file.write(image_data)
        temp_file_path = temp_file.name
    
    try:
        # Open the file as BufferedReader
        with open(temp_file_path, 'rb') as image_buffer:
            image_edit_optional_request_params = {}
            litellm_params = GenericLiteLLMParams()
            headers = {}

            data, files = image_edit_config.transform_image_edit_request(
                model=model,
                prompt=prompt,
                image=image_buffer,
                image_edit_optional_request_params=image_edit_optional_request_params,
                litellm_params=litellm_params,
                headers=headers,
            )

            # Check that data contains model and prompt but not image
            assert data["model"] == model
            assert data["prompt"] == prompt
            assert "image" not in data

            # Check that files contains the image with the original filename
            assert len(files) == 1
            assert files[0][0] == "image[]"
            # Should use the buffer's name (full path from the BufferedReader.name)
            assert files[0][1][0] == temp_file_path  # Uses full path from buffer.name
            assert files[0][1][1] == image_buffer  # Should be the buffer object
            # Content type detection defaults to PNG for fake data without image headers
            assert files[0][1][2].startswith("image/")  # Should detect some image type
    finally:
        # Clean up the temp file
        os.unlink(temp_file_path)


def test_transform_image_edit_request_with_optional_params(image_edit_config: OpenAIImageEditConfig):
    """Test transformation with optional parameters like size, quality, etc."""
    model = "dall-e-2"
    prompt = "Make the background blue"
    image = b"fake_image_data"
    image_edit_optional_request_params = {
        "size": "512x512",
        "response_format": "b64_json",
        "n": 2,
        "user": "test_user"
    }
    litellm_params = GenericLiteLLMParams()
    headers = {}

    data, files = image_edit_config.transform_image_edit_request(
        model=model,
        prompt=prompt,
        image=image,
        image_edit_optional_request_params=image_edit_optional_request_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    # Check that data contains all the optional parameters
    assert data["model"] == model
    assert data["prompt"] == prompt
    assert data["size"] == "512x512"
    assert data["response_format"] == "b64_json"
    assert data["n"] == 2
    assert data["user"] == "test_user"
    assert "image" not in data
    assert "mask" not in data

    # Check that files contains only the image
    assert len(files) == 1
    assert files[0][0] == "image[]"
    assert files[0][1][1] == image


def test_transform_image_edit_request_with_multiple_images(image_edit_config: OpenAIImageEditConfig):
    """Test transformation with multiple images and no mask"""
    model = "dall-e-2"
    prompt = "Make the background blue"
    image1 = b"fake_image_data_1"
    image2 = b"fake_image_data_2"
    image3 = b"fake_image_data_3"
    images = [image1, image2, image3]
    image_edit_optional_request_params = {"size": "1024x1024", "n": 1}
    litellm_params = GenericLiteLLMParams()
    headers = {}

    data, files = image_edit_config.transform_image_edit_request(
        model=model,
        prompt=prompt,
        image=images,
        image_edit_optional_request_params=image_edit_optional_request_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    # Check that data contains model, prompt, and optional params but not image or mask
    assert data["model"] == model
    assert data["prompt"] == prompt
    assert data["size"] == "1024x1024"
    assert data["n"] == 1
    assert "image" not in data
    assert "mask" not in data

    # Check that files contains all three images and no mask
    assert len(files) == 3
    
    # All files should be image entries with image[] key
    image_files = [f for f in files if f[0] == "image[]"]
    assert len(image_files) == 3
    
    # Check that all image data is present
    image_data_in_files = [f[1][1] for f in image_files]
    assert image1 in image_data_in_files
    assert image2 in image_data_in_files
    assert image3 in image_data_in_files
    
    # Check that all files have proper content type
    for file_entry in image_files:
        assert file_entry[1][0] == "image.png"  # filename
        assert file_entry[1][2].startswith("image/")  # content type


def test_transform_image_edit_request_with_mask_list(image_edit_config: OpenAIImageEditConfig):
    """Test transformation with mask as list (should take first element)"""
    model = "dall-e-2"
    prompt = "Make the background blue"
    image = b"fake_image_data"
    mask1 = b"fake_mask_data_1"
    mask2 = b"fake_mask_data_2"
    image_edit_optional_request_params = {"mask": [mask1, mask2]}
    litellm_params = GenericLiteLLMParams()
    headers = {}

    data, files = image_edit_config.transform_image_edit_request(
        model=model,
        prompt=prompt,
        image=image,
        image_edit_optional_request_params=image_edit_optional_request_params,
        litellm_params=litellm_params,
        headers=headers,
    )

    # Check that data contains model and prompt but not image or mask
    assert data["model"] == model
    assert data["prompt"] == prompt
    assert "image" not in data
    assert "mask" not in data

    # Check that files contains image and only the first mask
    assert len(files) == 2
    
    mask_file = next(f for f in files if f[0] == "mask")
    assert mask_file[1][1] == mask1  # Should be the first mask, not the second

