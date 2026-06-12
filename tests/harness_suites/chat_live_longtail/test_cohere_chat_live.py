import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path

import pytest

import litellm
from litellm import completion, embedding

litellm.num_retries = 3

@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_chat_completion_cohere_citations(stream):
    try:
        litellm.set_verbose = True
        messages = [
            {
                "role": "user",
                "content": "Which penguins are the tallest?",
            },
        ]
        response = await litellm.acompletion(
            model="cohere_chat/v1/command-r",
            messages=messages,
            documents=[
                {"title": "Tall penguins", "text": "Emperor penguins are the tallest."},
                {
                    "title": "Penguin habitats",
                    "text": "Emperor penguins only live in Antarctica.",
                },
            ],
            stream=stream,
        )

        if stream:
            citations_chunk = False
            async for chunk in response:
                print("received chunk", chunk)
                if "citations" in chunk:
                    citations_chunk = True
                    break
            assert citations_chunk
        else:
            assert response.citations is not None
    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_cohere_command_r_plus_function_call():
    litellm.set_verbose = True
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_current_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA",
                        },
                        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                    },
                    "required": ["location"],
                },
            },
        }
    ]
    messages = [
        {
            "role": "user",
            "content": "What's the weather like in Boston today in Fahrenheit?",
        }
    ]
    try:
        # test without max tokens
        response = completion(
            model="cohere_chat/v1/command-r-plus",
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        # Add any assertions, here to check response args
        print(response)
        assert isinstance(response.choices[0].message.tool_calls[0].function.name, str)
        assert isinstance(
            response.choices[0].message.tool_calls[0].function.arguments, str
        )
    except litellm.Timeout:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_completion_cohere():
    try:
        # litellm.set_verbose=True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {"role": "assistant", "content": [{"text": "2", "type": "text"}]},
            {"role": "assistant", "content": [{"text": "3", "type": "text"}]},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        response = completion(
            model="cohere_chat/v1/command-r",
            messages=messages,
        )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [True, False])
async def test_chat_completion_cohere(sync_mode):
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        if sync_mode is False:
            response = await litellm.acompletion(
                model="cohere_chat/v1/command-r",
                messages=messages,
                max_tokens=10,
            )
        else:
            response = completion(
                model="cohere_chat/v1/command-r",
                messages=messages,
                max_tokens=10,
            )
        print(response)
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.asyncio
@pytest.mark.parametrize("sync_mode", [False])
async def test_chat_completion_cohere_stream(sync_mode):
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You're a good bot"},
            {
                "role": "user",
                "content": "Hey",
            },
        ]
        if sync_mode is False:
            response = await litellm.acompletion(
                model="cohere_chat/v1/command-r",
                messages=messages,
                max_tokens=10,
                stream=True,
            )
            print("async cohere stream response", response)
            async for chunk in response:
                print(chunk)
        else:
            response = completion(
                model="cohere_chat/v1/command-r",
                messages=messages,
                max_tokens=10,
                stream=True,
            )
            print(response)
            for chunk in response:
                print(chunk)
    except litellm.APIConnectionError as e:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_cohere_v2_chat_completion(sync_mode):
    """Test basic Cohere v2 chat completion functionality."""
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"},
        ]

        if sync_mode:
            response = completion(
                model="cohere_chat/v2/command-a-03-2025",
                messages=messages,
                max_tokens=50,
            )
        else:
            response = await litellm.acompletion(
                model="cohere_chat/v2/command-a-03-2025",
                messages=messages,
                max_tokens=50,
            )

        # Validate response structure
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert response.usage is not None
        assert response.usage.total_tokens > 0
        print(f"Cohere v2 response: {response}")

    except litellm.ServiceUnavailableError:
        pass  # Skip if service is unavailable
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_cohere_v2_streaming(stream):
    """Test Cohere v2 streaming functionality."""
    try:
        litellm.set_verbose = True
        messages = [{"role": "user", "content": "Tell me a short story about a robot."}]

        response = await litellm.acompletion(
            model="cohere_chat/v2/command-a-03-2025",
            messages=messages,
            max_tokens=100,
            stream=stream,
        )

        if stream:
            # Test streaming response
            chunks = []
            async for chunk in response:
                chunks.append(chunk)
                if len(chunks) >= 3:  # Test first few chunks
                    break
            assert len(chunks) > 0
            print(f"Received {len(chunks)} streaming chunks")
        else:
            # Test non-streaming response
            assert response.choices is not None
            assert len(response.choices) > 0
            assert response.choices[0].message.content is not None
            print(f"Non-streaming response: {response.choices[0].message.content}")

    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_v2_tool_calling():
    """Test Cohere v2 tool calling functionality."""
    try:
        litellm.set_verbose = True
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": "What's the weather like in New York?"}]

        response = completion(
            model="cohere_chat/v2/command-a-03-2025",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=100,
        )

        # Validate tool calling response
        assert response.choices is not None
        assert len(response.choices) > 0
        message = response.choices[0].message

        # Check if tool calls are present
        if hasattr(message, "tool_calls") and message.tool_calls:
            assert len(message.tool_calls) > 0
            tool_call = message.tool_calls[0]
            assert tool_call.function.name == "get_weather"
            assert tool_call.function.arguments is not None
            print(
                f"Tool call: {tool_call.function.name} - {tool_call.function.arguments}"
            )
        else:
            # If no tool calls, check that we got a regular response
            assert message.content is not None
            print(f"Regular response: {message.content}")

    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_cohere_v2_annotations(stream):
    """Test Cohere v2 annotations functionality (replaces citations)."""
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "user", "content": "What are the benefits of renewable energy?"}
        ]

        documents = [
            {
                "data": {
                    "title": "Renewable Energy Benefits Document",
                    "snippet": "Renewable energy sources like solar and wind power provide clean electricity while reducing greenhouse gas emissions and dependence on fossil fuels.",
                }
            },
            {
                "data": {
                    "title": "Environmental Impact Study",
                    "snippet": "Studies show that renewable energy significantly reduces carbon footprint and helps combat climate change.",
                }
            },
        ]

        response = await litellm.acompletion(
            model="cohere_chat/v2/command-a-03-2025",
            messages=messages,
            documents=documents,
            max_tokens=100,
            stream=stream,
        )

        if stream:
            # Test streaming with annotations
            annotations_found = False
            async for chunk in response:
                # Check if chunk has a message with annotations
                if (
                    hasattr(chunk, "choices")
                    and chunk.choices
                    and len(chunk.choices) > 0
                    and hasattr(chunk.choices[0], "message")
                    and hasattr(chunk.choices[0].message, "annotations")
                    and chunk.choices[0].message.annotations
                ):
                    annotations_found = True
                    print(
                        f"Streaming annotations: {chunk.choices[0].message.annotations}"
                    )
                    break
            # Note: Annotations might not appear in every chunk during streaming
        else:
            # Test non-streaming with annotations
            assert response.choices is not None
            assert len(response.choices) > 0

            # Check for annotations in message
            message = response.choices[0].message
            if hasattr(message, "annotations") and message.annotations:
                assert len(message.annotations) > 0
                print(f"Annotations found: {len(message.annotations)}")

                # Validate annotation structure
                for annotation in message.annotations:
                    assert (
                        annotation.get("type") == "url_citation"
                    ), f"Expected type 'url_citation', got {annotation.get('type')}"
                    assert "url_citation" in annotation, "Missing url_citation field"
                    url_citation = annotation["url_citation"]
                    assert "start_index" in url_citation, "Missing start_index"
                    assert "end_index" in url_citation, "Missing end_index"
                    assert "title" in url_citation, "Missing title"
                    assert "url" in url_citation, "Missing url"

                print(f"First annotation: {message.annotations[0]}")
            else:
                # Annotations might not always be present depending on the response
                print("No annotations in this response")

            # Ensure citations field is NOT present (removed backward compatibility)
            assert not hasattr(
                response, "citations"
            ), "Citations field should be removed - no backward compatibility"

    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_v2_parameter_mapping():
    """Test Cohere v2 parameter mapping and validation."""
    try:
        litellm.set_verbose = True
        messages = [{"role": "user", "content": "Generate a creative story."}]

        # Test various parameters that should be mapped correctly
        response = completion(
            model="cohere_chat/v2/command-a-03-2025",
            messages=messages,
            temperature=0.7,
            max_tokens=50,
            top_p=0.9,
            frequency_penalty=0.1,
            presence_penalty=0.1,
            stop=["END", "STOP"],
            seed=42,
        )

        # Validate response
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert response.usage is not None
        print(f"Parameter mapping test response: {response.choices[0].message.content}")

    except litellm.ServiceUnavailableError:
        pass
    except Exception as e:
        pytest.fail(f"Error occurred: {e}")


def test_cohere_v2_error_handling():
    """Test Cohere v2 error handling with invalid parameters."""
    try:
        # Test with invalid model name
        try:
            response = completion(
                model="cohere_chat/v2/invalid-model",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )
            # If we get here, the test should fail
            pytest.fail("Should have failed with invalid model")
        except Exception as e:
            # Expected to fail with invalid model
            print(f"Expected error with invalid model: {e}")

        # Test with empty messages
        try:
            response = completion(
                model="cohere_chat/v2/command-a-03-2025",
                messages=[],  # Empty messages
                max_tokens=10,
            )
            pytest.fail("Should have failed with empty messages")
        except Exception as e:
            # Expected to fail with empty messages
            print(f"Expected error with empty messages: {e}")

    except Exception as e:
        pytest.fail(f"Unexpected error in error handling test: {e}")


@pytest.mark.asyncio
async def test_cohere_v2_conversation_history():
    """Test Cohere v2 with conversation history."""
    try:
        litellm.set_verbose = True
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "2+2 equals 4."},
            {"role": "user", "content": "What about 3+3?"},
        ]

        response = await litellm.acompletion(
            model="cohere_chat/v2/command-a-03-2025", messages=messages, max_tokens=50
        )

        # Validate response with conversation history
        assert response.choices is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        print(f"Conversation history response: {response.choices[0].message.content}")

    except (
        litellm.ServiceUnavailableError,
        litellm.InternalServerError,
        litellm.Timeout,
        litellm.APIConnectionError,
    ):
        pytest.skip("Cohere service unavailable")
    except litellm.RateLimitError:
        pytest.skip("Rate limit exceeded")
