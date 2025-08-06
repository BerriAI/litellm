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
            "model": "gemini/gemini-2.5-flash-lite",
        }

@pytest.mark.asyncio
async def test_mock_stream_generate_content_with_tools():
    """Test streaming function call response parsing and validation"""
    from litellm.types.google_genai.main import ToolConfigDict
    litellm._turn_on_debug()
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
            model="gemini/gemini-2.5-flash-lite",
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

@pytest.mark.asyncio
async def test_validate_post_request_parameters():
    """
    Test that the correct parameters are sent in the POST request to Google GenAI API
    
    Params validated
        1. model
        2. contents
        3. tools
    """
    from litellm.types.google_genai.main import ToolConfigDict
    
    contents = [
        {
            "role": "user",
            "parts": [
                {"text": "Schedule a meeting with Bob and Alice for 03/27/2025 at 10:00 AM about the Q3 planning"}
            ]
        }
    ]
    
    tools = [
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

    # Mock response for the HTTP request
    raw_chunks = [
        b"data: [DONE]\n\n"
    ]

    # Mock the HTTP handler to capture the request
    with unittest.mock.patch("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", new_callable=unittest.mock.AsyncMock) as mock_post:
        # Create mock response object
        mock_response = unittest.mock.MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {"content-type": "application/json"}
        
        # Mock the aiter_bytes method
        async def mock_aiter_bytes():
            for chunk in raw_chunks:
                yield chunk
        
        mock_response.aiter_bytes = mock_aiter_bytes
        mock_post.return_value = mock_response

        print("\n--- Testing POST request parameters validation ---")
        
        # Make the API call
        response = await litellm.google_genai.agenerate_content_stream(
            model="gemini/gemini-2.5-flash-lite",
            contents=contents,
            tools=tools
        )
        
        # Consume the response to ensure the request is made
        async for chunk in response:
            pass
        
        # Validate that the HTTP post was called
        assert mock_post.called, "Expected HTTP POST to be called"
        
        # Get the call arguments
        call_args, call_kwargs = mock_post.call_args
        
        print(f"POST call args: {call_args}")
        print(f"POST call kwargs: {call_kwargs}")
        
        # Validate URL contains the correct endpoint
        if call_args:
            url = call_args[0] if len(call_args) > 0 else call_kwargs.get('url')
            assert url is not None, "Expected URL to be provided"
            assert "generativelanguage.googleapis.com" in url, f"Expected Google API URL, got: {url}"
            assert "streamGenerateContent" in url, f"Expected streamGenerateContent endpoint, got: {url}"
            print(f"✅ URL validation passed: {url}")
        
        # Get the request data/json from the call
        request_data = None
        if 'data' in call_kwargs:
            # If data is passed as bytes, decode it
            if isinstance(call_kwargs['data'], bytes):
                request_data = json.loads(call_kwargs['data'].decode('utf-8'))
            else:
                request_data = call_kwargs['data']
        elif 'json' in call_kwargs:
            request_data = call_kwargs['json']
        
        assert request_data is not None, "Expected request data to be provided"
        print(f"Request data: {json.dumps(request_data, indent=2)}")
        
        # Validate model field
        assert "model" in request_data, "Expected 'model' field in request data"
        # Model might be transformed, but should contain gemini-2.5-flash-lite
        model_value = request_data["model"]
        assert "gemini-2.5-flash-lite" in model_value, f"Expected model to contain 'gemini-2.5-flash-lite', got: {model_value}"
        print(f"✅ Model validation passed: {model_value}")
        
        # Validate contents field
        assert "contents" in request_data, "Expected 'contents' field in request data"
        request_contents = request_data["contents"]
        assert isinstance(request_contents, list), "Expected contents to be a list"
        assert len(request_contents) > 0, "Expected at least one content item"
        
        # Check the first content item
        first_content = request_contents[0]
        assert "role" in first_content, "Expected 'role' in content item"
        assert first_content["role"] == "user", f"Expected role 'user', got: {first_content['role']}"
        assert "parts" in first_content, "Expected 'parts' in content item"
        assert isinstance(first_content["parts"], list), "Expected parts to be a list"
        assert len(first_content["parts"]) > 0, "Expected at least one part"
        
        # Check the text content
        first_part = first_content["parts"][0]
        assert "text" in first_part, "Expected 'text' in part"
        expected_text = "Schedule a meeting with Bob and Alice for 03/27/2025 at 10:00 AM about the Q3 planning"
        assert first_part["text"] == expected_text, f"Expected text '{expected_text}', got: {first_part['text']}"
        print(f"✅ Contents validation passed")
        
        # Validate tools field
        assert "tools" in request_data, "Expected 'tools' field in request data"
        request_tools = request_data["tools"]
        assert isinstance(request_tools, list), "Expected tools to be a list"
        assert len(request_tools) > 0, "Expected at least one tool"
        
        # Check the first tool
        first_tool = request_tools[0]
        assert "functionDeclarations" in first_tool, "Expected 'functionDeclarations' in tool"
        function_declarations = first_tool["functionDeclarations"]
        assert isinstance(function_declarations, list), "Expected functionDeclarations to be a list"
        assert len(function_declarations) > 0, "Expected at least one function declaration"
        
        # Check the function declaration
        func_decl = function_declarations[0]
        assert "name" in func_decl, "Expected 'name' in function declaration"
        assert func_decl["name"] == "schedule_meeting", f"Expected function name 'schedule_meeting', got: {func_decl['name']}"
        assert "description" in func_decl, "Expected 'description' in function declaration"
        assert "parameters" in func_decl, "Expected 'parameters' in function declaration"
        
        # Check function parameters
        params = func_decl["parameters"]
        assert "type" in params, "Expected 'type' in parameters"
        assert params["type"] == "object", f"Expected parameters type 'object', got: {params['type']}"
        assert "properties" in params, "Expected 'properties' in parameters"
        assert "required" in params, "Expected 'required' in parameters"
        
        # Check required fields
        required_fields = params["required"]
        expected_required = ["attendees", "date", "time", "topic"]
        assert set(required_fields) == set(expected_required), f"Expected required fields {expected_required}, got: {required_fields}"
        print(f"✅ Tools validation passed")
        
        print("✅ All POST request parameter validations passed!")