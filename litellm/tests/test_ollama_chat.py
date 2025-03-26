import pytest

import litellm
from unittest.mock import MagicMock, patch


def test_ollama_chat_image():
    """
    Test that datauri prefixes are removed, JPEG/PNG images are passed
    through, and other image formats are converted to JPEG.  Non-image
    data is untouched.
    """

    import base64
    import io

    from PIL import Image

    def mock_post(url, **kwargs):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.json.return_value = {
            "model": "llama3.2-vision:11b",
            "created_at": "2024-11-23T13:16:28.5525725Z",
            "message": {
                "role": "assistant",
                "content": "The image is a blank",
                "images": kwargs["json"]["messages"][0]["images"]
            },
            "done_reason": "stop",
            "done": True,
            "total_duration": 74458830900,
            "load_duration": 18295722500,
            "prompt_eval_count": 17,
            "prompt_eval_duration": 9979000000,
            "eval_count": 104,
            "eval_duration": 46036000000,
        }
        return mock_response

    def make_b64image(format):
        image = Image.new(mode="RGB", size=(1, 1))
        image_buffer = io.BytesIO()
        image.save(image_buffer, format)
        return base64.b64encode(image_buffer.getvalue()).decode("utf-8")

    jpeg_image = make_b64image("JPEG")
    webp_image = make_b64image("WEBP")
    png_image = make_b64image("PNG")

    base64_data = base64.b64encode(b"some random data")
    datauri_base64_data = f"data:text/plain;base64,{base64_data}"

    tests = [
        # input                                    expected
        [jpeg_image, jpeg_image],
        [webp_image, None],
        [png_image, png_image],
        [f"data:image/jpeg;base64,{jpeg_image}", jpeg_image],
        [f"data:image/webp;base64,{webp_image}", None],
        [f"data:image/png;base64,{png_image}", png_image],
        [datauri_base64_data, datauri_base64_data],
    ]

    for test in tests:
        try:
            with patch("requests.post", side_effect=mock_post):
                response = litellm.completion(
                    model="ollama_chat/llava",
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Whats in this image?"},
                                {
                                    "type": "image_url",
                                    "image_url": {"url": test[0]},
                                },
                            ],
                        }
                    ],
                )
                if not test[1]:
                    # the conversion process may not always generate the same image,
                    # so just check for a JPEG image when a conversion was done.
                    image_data = response["choices"][0]["message"]["images"][0]
                    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
                    assert image.format == "JPEG"
                else:
                    assert response["choices"][0]["message"]["images"][0] == test[1]
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")
