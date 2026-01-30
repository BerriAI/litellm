"""
Test that stream_chunk_builder correctly preserves images from streaming chunks.

This tests the fix for https://github.com/BerriAI/litellm/issues/19478
where images from models like gemini-2.5-flash-image were lost when
rebuilding the response from streaming chunks.
"""
import pytest
import litellm
from litellm import stream_chunk_builder


def test_stream_chunk_builder_preserves_images():
    """
    Test that stream_chunk_builder correctly preserves images from streaming chunks.
    """
    # Simulate streaming chunks from an image generation model
    init_chunks = [
        {
            "id": "chatcmpl-image-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                    },
                    "finish_reason": None,
                }
            ],
            "created": 1737654321,
            "model": "gemini/gemini-2.5-flash-image",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-image-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "images": [
                            {
                                "image_url": {
                                    "url": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
                                    "detail": "auto"
                                },
                                "index": 0,
                                "type": "image_url"
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
            "created": 1737654321,
            "model": "gemini/gemini-2.5-flash-image",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-image-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "created": 1737654321,
            "model": "gemini/gemini-2.5-flash-image",
            "object": "chat.completion.chunk",
        },
    ]

    chunks = []
    for chunk in init_chunks:
        chunks.append(litellm.ModelResponse(**chunk, stream=True))

    response = stream_chunk_builder(chunks=chunks)

    # Verify that images are preserved in the rebuilt response
    assert response.choices[0].message.images is not None, "Images should be preserved in stream_chunk_builder"
    assert len(response.choices[0].message.images) == 1, "Should have exactly 1 image"
    assert response.choices[0].message.images[0]["type"] == "image_url"
    assert "base64" in response.choices[0].message.images[0]["image_url"]["url"]


def test_stream_chunk_builder_preserves_multiple_images():
    """
    Test that stream_chunk_builder correctly preserves multiple images from different chunks.
    """
    init_chunks = [
        {
            "id": "chatcmpl-multi-image-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "Here are your images:",
                    },
                    "finish_reason": None,
                }
            ],
            "created": 1737654321,
            "model": "gemini/gemini-2.5-flash-image",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-multi-image-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "images": [
                            {
                                "image_url": {"url": "data:image/png;base64,image1data", "detail": "auto"},
                                "index": 0,
                                "type": "image_url"
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
            "created": 1737654321,
            "model": "gemini/gemini-2.5-flash-image",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-multi-image-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "images": [
                            {
                                "image_url": {"url": "data:image/png;base64,image2data", "detail": "auto"},
                                "index": 1,
                                "type": "image_url"
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
            "created": 1737654321,
            "model": "gemini/gemini-2.5-flash-image",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-multi-image-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "created": 1737654321,
            "model": "gemini/gemini-2.5-flash-image",
            "object": "chat.completion.chunk",
        },
    ]

    chunks = []
    for chunk in init_chunks:
        chunks.append(litellm.ModelResponse(**chunk, stream=True))

    response = stream_chunk_builder(chunks=chunks)

    # Verify content is preserved
    assert response.choices[0].message.content == "Here are your images:"

    # Verify all images are preserved
    assert response.choices[0].message.images is not None, "Images should be preserved"
    assert len(response.choices[0].message.images) == 2, "Should have exactly 2 images"
    assert "image1data" in response.choices[0].message.images[0]["image_url"]["url"]
    assert "image2data" in response.choices[0].message.images[1]["image_url"]["url"]


def test_stream_chunk_builder_no_images():
    """
    Test that stream_chunk_builder works correctly when there are no images (regression test).
    """
    init_chunks = [
        {
            "id": "chatcmpl-no-image-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": "Hello, ",
                    },
                    "finish_reason": None,
                }
            ],
            "created": 1737654321,
            "model": "gpt-4",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-no-image-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "content": "world!",
                    },
                    "finish_reason": None,
                }
            ],
            "created": 1737654321,
            "model": "gpt-4",
            "object": "chat.completion.chunk",
        },
        {
            "id": "chatcmpl-no-image-test",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }
            ],
            "created": 1737654321,
            "model": "gpt-4",
            "object": "chat.completion.chunk",
        },
    ]

    chunks = []
    for chunk in init_chunks:
        chunks.append(litellm.ModelResponse(**chunk, stream=True))

    response = stream_chunk_builder(chunks=chunks)

    # Verify content is preserved
    assert response.choices[0].message.content == "Hello, world!"

    # Verify images attribute doesn't exist or is None (no images in this stream)
    images = getattr(response.choices[0].message, 'images', None)
    assert images is None, "Should not have images when none were in the stream"
