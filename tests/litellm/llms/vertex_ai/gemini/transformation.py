import json
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(
    0, os.path.abspath("../../../../..")
)  # Adds the parent directory to the system path

from litellm.llms.vertex_ai.gemini.transformation import (
    _gemini_convert_messages_with_history,
)

@pytest.mark.parametrize(
    "media_type, mime_type, content",
    [
        ("image_url", "image/jpeg", "1234abcd"),
        ("video_url", "video/mp4", "dcba4321"),
    ],
)
def test_vertex_only_media_user_message_with_base64data(media_type: str, mime_type: str, content: str):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": media_type,
                    media_type: {"url": f"data:{mime_type};base64,{content}"},
                },
            ],
        },
    ]

    response = _gemini_convert_messages_with_history(messages=messages)
    print("response", response)

    expected_response = [
        {
            "role": "user",
            "parts": [
                {
                    "inline_data": {
                        "data": content,
                        "mime_type": mime_type,
                    }
                },
                {"text": " "},
            ],
        }
    ]

    assert len(response) == len(expected_response)
    for idx, content in enumerate(response):
        assert (
            content == expected_response[idx]
        ), "Invalid gemini input. Got={}, Expected={}".format(
            content, expected_response[idx]
        )


@pytest.mark.parametrize(
    "media_type, mime_type, content",
    [
        ("image_url", "image/jpeg", "https://i.pinimg.com/736x/b4/b1/be/b4b1becad04d03a9071db2817fc9fe77.jpg"),
        ("video_url", "video/mp4", "https://videos.pexels.com/video-files/3571264/3571264-sd_426_240_30fps.mp4"),
    ],
)
def test_vertex_only_media_user_message_with_url(media_type: str, mime_type: str, content: str):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": media_type,
                    media_type: content,
                },
            ],
        },
    ]

    response = _gemini_convert_messages_with_history(messages=messages)
    print("response", response)

    expected_response = [
        {
            "role": "user",
            "parts": [
                {
                    "file_data": {
                        "file_uri": content,
                        "mime_type": mime_type,
                    }
                },
                {"text": " "},
            ],
        }
    ]

    assert len(response) == len(expected_response)
    for idx, content in enumerate(response):
        assert (
            content == expected_response[idx]
        ), "Invalid gemini input. Got={}, Expected={}".format(
            content, expected_response[idx]
        )
