import pytest
import requests
import base64
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PROXY_URL = "http://0.0.0.0:4000/chat/completions"

def create_base64_pdf():
    """Create a minimal valid PDF in base64 format"""
    minimal_pdf = b"%PDF-1.0\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n3 0 obj<</Type/Page/MediaBox[0 0 3 3]>>endobj\nxref\n0 4\n0000000000 65535 f\n0000000010 00000 n\n0000000053 00000 n\n0000000102 00000 n\ntrailer<</Size 4/Root 1 0 R>>\nstartxref\n149\n%EOF"
    return base64.b64encode(minimal_pdf).decode()

def create_base64_png():
    """Create a simple PNG in base64 format (red dot)"""
    return "iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg=="

def create_base64_jpeg():
    """Create a simple JPEG in base64 format (white pixel)"""
    return "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigD//2Q=="

def test_pdf_content_raises_400():
    """Test that sending PDF content raises a 400 error with correct message"""
    pdf_base64 = create_base64_pdf()
    
    payload = {
        "model": "claude-3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:application/pdf;base64,{pdf_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "What's in this document?"
                    }
                ]
            }
        ]
    }

    response = requests.post(
        PROXY_URL,
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 400
    response_data = response.json()
    
    logger.info(f"Error response for PDF test: {response_data}")
    
    error_message = response_data["error"]["message"]
    assert "litellm.BadRequestError: AnthropicException" in error_message
    assert "claude-3-haiku-20240307" in error_message
    assert "does not support PDF input" in error_message

def test_png_content_succeeds():
    """Test that sending PNG content works successfully"""
    png_base64 = create_base64_png()
    
    payload = {
        "model": "claude-3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{png_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    }
                ]
            }
        ]
    }

    response = requests.post(
        PROXY_URL,
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    response_data = response.json()
    logger.info(f"Successful PNG response status: {response.status_code}")
    
    # Check response has expected structure
    assert "choices" in response_data
    assert len(response_data["choices"]) > 0
    assert "message" in response_data["choices"][0]
    assert "content" in response_data["choices"][0]["message"]

def test_jpeg_content_succeeds():
    """Test that sending JPEG content works successfully"""
    jpeg_base64 = create_base64_jpeg()
    
    payload = {
        "model": "claude-3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{jpeg_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "What's in this image?"
                    }
                ]
            }
        ]
    }

    response = requests.post(
        PROXY_URL,
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    response_data = response.json()
    logger.info(f"Successful JPEG response status: {response.status_code}")
    
    # Check response has expected structure
    assert "choices" in response_data
    assert len(response_data["choices"]) > 0
    assert "message" in response_data["choices"][0]
    assert "content" in response_data["choices"][0]["message"]

def test_multiple_images_succeeds():
    """Test that sending multiple supported images in one request works"""
    png_base64 = create_base64_png()
    jpeg_base64 = create_base64_jpeg()
    
    payload = {
        "model": "claude-3",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{png_base64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{jpeg_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": "What's in these images?"
                    }
                ]
            }
        ]
    }

    response = requests.post(
        PROXY_URL,
        json=payload,
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    response_data = response.json()
    logger.info(f"Successful multiple images response status: {response.status_code}")
    
    # Check response has expected structure
    assert "choices" in response_data
    assert len(response_data["choices"]) > 0
    assert "message" in response_data["choices"][0]
    assert "content" in response_data["choices"][0]["message"]