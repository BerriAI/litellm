from base_google_test import BaseGoogleGenAITest
import sys
import os
sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
import unittest.mock
import json

class TestGoogleGenAIStudio(BaseGoogleGenAITest):
    """Test Google GenAI Studio"""

    @property
    def model_config(self):
        return {
            "model": "gemini/gemini-1.5-flash",
        }

@pytest.mark.asyncio
async def test_mock_stream_generate_content_with_tools():
    """Test streaming function call response parsing and validation"""
    from litellm.types.google_genai.main import ToolConfigDict
    
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": "Schedule a meeting with Bob and Alice for 03/27/2025 at 10:00 AM about the Q3 planning"}
            ]
        }
    ]

    # Mock streaming response chunks that represent a function call response
    mock_response_chunk = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "functionCall": {
                                "name": "schedule_meeting",
                                "args": {
                                    "attendees": ["Bob", "Alice"],
                                    "date": "2025-03-27",
                                    "time": "10:00",
                                    "topic": "Q3 planning"
                                }
                            }
                        }
                    ],
                    "role": "model"
                },
                "finishReason": "STOP",
                "index": 0
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 15,
            "candidatesTokenCount": 5,
            "totalTokenCount": 20
        }
    }

    # Convert to bytes as expected by the streaming iterator
    raw_chunks = [
        f"data: {json.dumps(mock_response_chunk)}\n\n".encode(),
        b"data: [DONE]\n\n"
    ]

    # Mock the HTTP handler
    with unittest.mock.patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=unittest.mock.AsyncMock) as mock_post:
        # Create mock response object
        mock_response = unittest.mock.MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        
        # Mock the aiter_bytes method to return our chunks as bytes
        async def mock_aiter_bytes():
            for chunk in raw_chunks:
                yield chunk
        
        mock_response.aiter_bytes = mock_aiter_bytes
        mock_post.return_value = mock_response

        print("\n--- Testing async agenerate_content_stream with function call parsing ---")
        response = await litellm.google_genai.agenerate_content_stream(
            model="gemini/gemini-1.5-flash",
            contents=contents,
            tools=[
                {
                    "functionDeclarations": [
                        {
                            "name": "schedule_meeting",
                            "description": "Schedules a meeting with specified attendees at a given time and date.",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "attendees": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "List of people attending the meeting."
                                    },
                                    "date": {
                                        "type": "string",
                                        "description": "Date of the meeting (e.g., '2024-07-29')"
                                    },
                                    "time": {
                                        "type": "string",
                                        "description": "Time of the meeting (e.g., '15:00')"
                                    },
                                    "topic": {
                                        "type": "string",
                                        "description": "The subject or topic of the meeting."
                                    }
                                },
                                "required": ["attendees", "date", "time", "topic"]
                            }
                        }
                    ]
                }
            ]
        )
        
        # Collect all chunks and parse function calls
        chunks = []
        function_calls = []
        
        chunk_count = 0
        async for chunk in response:
            chunk_count += 1
            print(f"Received chunk {chunk_count}: {chunk}")
            chunks.append(chunk)
            
            # Stop after a reasonable number of chunks to prevent infinite loop
            if chunk_count > 10:
                break
            
            # Parse function calls from byte chunks
            if isinstance(chunk, bytes):
                try:
                    # Decode bytes to string
                    chunk_str = chunk.decode('utf-8')
                    print(f"Decoded chunk: {chunk_str}")
                    
                    # Extract JSON from Server-Sent Events format (data: {...})
                    if chunk_str.startswith('data: ') and not chunk_str.startswith('data: [DONE]'):
                        json_str = chunk_str[6:].strip()  # Remove 'data: ' prefix
                        try:
                            parsed_json = json.loads(json_str)
                            print(f"Parsed JSON: {parsed_json}")
                            
                            # Parse function calls from the JSON
                            if "candidates" in parsed_json:
                                for candidate in parsed_json["candidates"]:
                                    if "content" in candidate and "parts" in candidate["content"]:
                                        for part in candidate["content"]["parts"]:
                                            if "functionCall" in part:
                                                function_calls.append({
                                                    'name': part["functionCall"]["name"],
                                                    'args': part["functionCall"]["args"]
                                                })
                                                print(f"Found function call: {part['functionCall']}")
                        except json.JSONDecodeError as e:
                            print(f"Failed to parse JSON: {e}")
                except UnicodeDecodeError as e:
                    print(f"Failed to decode bytes: {e}")
            
            # Handle dict responses (in case some chunks are already parsed)
            elif isinstance(chunk, dict):
                # Direct dict response
                if "candidates" in chunk:
                    for candidate in chunk["candidates"]:
                        if "content" in candidate and "parts" in candidate["content"]:
                            for part in candidate["content"]["parts"]:
                                if "functionCall" in part:
                                    function_calls.append({
                                        'name': part["functionCall"]["name"],
                                        'args': part["functionCall"]["args"]
                                    })
            
            # Handle object responses with attributes
            elif hasattr(chunk, 'candidates') and chunk.candidates:
                for candidate in chunk.candidates:
                    if hasattr(candidate, 'content') and candidate.content:
                        if hasattr(candidate.content, 'parts') and candidate.content.parts:
                            for part in candidate.content.parts:
                                if hasattr(part, 'function_call') and part.function_call:
                                    function_calls.append({
                                        'name': part.function_call.name,
                                        'args': part.function_call.args
                                    })
        
        # Assertions
        print(f"\nFunction calls found: {function_calls}")
        print(f"Total chunks received: {chunk_count}")
        
        # Assert we found at least one function call
        assert len(function_calls) > 0, "Expected at least one function call in the streaming response"
        
        # Check the first function call
        function_call = function_calls[0]
        
        # Assert function name
        assert function_call['name'] == "schedule_meeting", f"Expected function name 'schedule_meeting', got '{function_call['name']}'"
        
        # Assert function arguments
        args = function_call['args']
        assert "attendees" in args, "Expected 'attendees' in function call arguments"
        assert "date" in args, "Expected 'date' in function call arguments"
        assert "time" in args, "Expected 'time' in function call arguments"
        assert "topic" in args, "Expected 'topic' in function call arguments"
        
        # Assert specific argument values
        assert args["attendees"] == ["Bob", "Alice"], f"Expected attendees ['Bob', 'Alice'], got {args['attendees']}"
        assert args["date"] == "2025-03-27", f"Expected date '2025-03-27', got {args['date']}"
        assert args["time"] == "10:00", f"Expected time '10:00', got {args['time']}"
        assert args["topic"] == "Q3 planning", f"Expected topic 'Q3 planning', got {args['topic']}"
        
        print("✅ All function call assertions passed!")