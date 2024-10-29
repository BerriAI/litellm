import json
import os
import sys
import traceback

from dotenv import load_dotenv

load_dotenv()
import io
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import litellm


def test_completion_pydantic_obj_2():
    from pydantic import BaseModel
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    litellm.set_verbose = True

    class CalendarEvent(BaseModel):
        name: str
        date: str
        participants: list[str]

    class EventsList(BaseModel):
        events: list[CalendarEvent]

    messages = [
        {"role": "user", "content": "List important events from the 20th century."}
    ]
    expected_request_body = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": "List important events from the 20th century."}],
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "response_schema": {
                "properties": {
                    "events": {
                        "items": {
                            "properties": {
                                "name": {"type": "string"},
                                "date": {"type": "string"},
                                "participants": {
                                    "items": {"type": "string"},
                                    "type": "array",
                                },
                            },
                            "required": [
                                "name",
                                "date",
                                "participants",
                            ],
                            "type": "object",
                        },
                        "type": "array",
                    }
                },
                "required": [
                    "events",
                ],
                "type": "object",
            },
        },
    }
    client = HTTPHandler()
    with patch.object(client, "post", new=MagicMock()) as mock_post:
        mock_post.return_value = expected_request_body
        try:
            litellm.completion(
                model="gemini/gemini-1.5-pro",
                messages=messages,
                response_format=EventsList,
                client=client,
            )
        except Exception as e:
            print(e)

        mock_post.assert_called_once()

        print(mock_post.call_args.kwargs)

        assert mock_post.call_args.kwargs["json"] == expected_request_body


def test_build_vertex_schema():
    from litellm.llms.vertex_ai_and_google_ai_studio.common_utils import (
        _build_vertex_schema,
    )
    import json

    schema = {
        "type": "object",
        "properties": {
            "recipes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"recipe_name": {"type": "string"}},
                    "required": ["recipe_name"],
                },
            }
        },
        "required": ["recipes"],
    }

    new_schema = _build_vertex_schema(schema)
    print(f"new_schema: {new_schema}")
    assert new_schema["type"] == schema["type"]
    assert new_schema["properties"] == schema["properties"]
    assert "required" in new_schema and new_schema["required"] == schema["required"]
