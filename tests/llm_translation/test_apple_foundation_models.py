"""
Integration tests for Apple Foundation Models provider.

Note: These tests will only pass on macOS 26.0+ with Apple Intelligence enabled.
"""

import os
import sys
import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock

sys.path.insert(0, os.path.abspath("../.."))

import litellm
from litellm import completion, acompletion
from pydantic import BaseModel

import litellm.llms.apple_foundation_models.chat.transformation
import litellm.llms.apple_foundation_models.common_utils

pytestmark = pytest.mark.filterwarnings("ignore:Pydantic.*")

try:
    from applefoundationmodels import apple_intelligence_available
except Exception:  # pragma: no cover - SDK absent

    def apple_intelligence_available() -> bool:
        return False


class TestAppleFoundationModelsProvider:
    """Test suite for Apple Foundation Models provider."""

    @pytest.fixture
    def mock_session_class(self):
        """Create a mock Session class."""
        mock_session = Mock()
        mock_response = Mock()
        mock_response.text = "This is a test response."
        mock_response.tool_calls = None
        mock_session.generate.return_value = mock_response

        mock_session_cls = Mock(return_value=mock_session)
        return mock_session_cls

    @pytest.fixture
    def mock_async_session_class(self):
        """Create a mock AsyncSession class."""
        mock_session = Mock()

        async def mock_stream_generator():
            chunks_text = ["This ", "is ", "a ", "test ", "response."]
            for text in chunks_text:
                chunk = Mock()
                chunk.content = text
                chunk.finish_reason = None
                yield chunk

        def mock_generate(*args, **kwargs):
            if kwargs.get("stream", False):
                return mock_stream_generator()

            async def async_response():
                mock_response = Mock()
                mock_response.text = "Async response"
                mock_response.tool_calls = None
                return mock_response

            return async_response()

        mock_session.generate = mock_generate

        mock_session_cls = Mock(return_value=mock_session)
        return mock_session_cls

    def test_import_error_handling(self):
        """Test that appropriate error is raised when package is not installed."""
        with patch(
            "litellm.llms.apple_foundation_models.chat.transformation.get_apple_session_class"
        ) as mock_get_session:
            mock_get_session.side_effect = ImportError(
                "Missing apple-foundation-models package. This is required for the "
                "Apple Foundation Models provider. Install it with: "
                "pip install apple-foundation-models"
            )

            with pytest.raises(Exception) as exc_info:
                completion(
                    model="apple_foundation_models/system",
                    messages=[{"role": "user", "content": "Hello"}],
                )

            error_message = str(exc_info.value)
            assert (
                "apple-foundation-models" in error_message
                or "Missing apple-foundation-models" in error_message
            )

    def test_availability_check_failure(self):
        """Test that appropriate error is raised when Apple Intelligence is not available."""
        with patch(
            "litellm.llms.apple_foundation_models.chat.transformation.get_apple_session_class"
        ) as mock_get_session:
            mock_get_session.side_effect = RuntimeError(
                "Apple Intelligence is not available on this system. "
                "Reason: macOS version too old. "
                "Requirements: macOS 26.0+ (Sequoia) with Apple Intelligence enabled."
            )

            with pytest.raises(Exception) as exc_info:
                completion(
                    model="apple_foundation_models/system",
                    messages=[{"role": "user", "content": "Hello"}],
                )

            error_message = str(exc_info.value)
            assert (
                "Apple Intelligence" in error_message
                or "macOS version" in error_message
            )

    @patch(
        "litellm.llms.apple_foundation_models.chat.transformation.get_apple_session_class"
    )
    def test_basic_completion(self, mock_get_session_func, mock_session_class):
        """Test basic completion without streaming."""
        mock_get_session_func.return_value = mock_session_class

        response = completion(
            model="apple_foundation_models/system",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content == "This is a test response."
        assert response.choices[0].message.role == "assistant"

    @patch(
        "litellm.llms.apple_foundation_models.chat.transformation.get_apple_session_class"
    )
    def test_completion_with_temperature(
        self, mock_get_session_func, mock_session_class
    ):
        """Test completion with custom temperature parameter."""
        mock_get_session_func.return_value = mock_session_class

        response = completion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "Tell me a joke"}],
            temperature=0.7,
        )

        mock_session = mock_session_class.return_value
        call_kwargs = mock_session.generate.call_args[1]
        assert call_kwargs["temperature"] == 0.7

    @patch(
        "litellm.llms.apple_foundation_models.chat.transformation.get_apple_session_class"
    )
    def test_completion_with_max_tokens(
        self, mock_get_session_func, mock_session_class
    ):
        """Test completion with max_tokens parameter."""
        mock_get_session_func.return_value = mock_session_class

        response = completion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "Write a story"}],
            max_tokens=2048,
        )

        mock_session = mock_session_class.return_value
        call_kwargs = mock_session.generate.call_args[1]
        assert call_kwargs["max_tokens"] == 2048

    @pytest.mark.asyncio
    @patch(
        "litellm.llms.apple_foundation_models.chat.transformation.get_apple_async_session_class"
    )
    async def test_async_streaming_completion(
        self, mock_get_async_session_func, mock_async_session_class
    ):
        """Test async streaming completion."""
        mock_get_async_session_func.return_value = mock_async_session_class

        response = await acompletion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "Count to 5"}],
            stream=True,
        )

        chunks = []
        async for chunk in response:
            if hasattr(chunk, "choices") and len(chunk.choices) > 0:
                if chunk.choices[0].delta.content:
                    chunks.append(chunk.choices[0].delta.content)

        assert len(chunks) > 0
        full_response = "".join(chunks)
        assert "test response" in full_response

    @patch(
        "litellm.llms.apple_foundation_models.chat.transformation.get_apple_session_class"
    )
    def test_completion_with_system_message(
        self, mock_get_session_func, mock_session_class
    ):
        """Test that system messages are properly handled."""
        mock_get_session_func.return_value = mock_session_class

        system_content = "You are a pirate. Speak like one."

        response = completion(
            model="apple_foundation_models/system",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": "Hello"},
            ],
        )

        create_session_kwargs = mock_session_class.call_args[1]
        assert create_session_kwargs["instructions"] == system_content

    @patch(
        "litellm.llms.apple_foundation_models.chat.transformation.get_apple_session_class"
    )
    def test_completion_with_tools(self, mock_get_session_func, mock_session_class):
        """Test completion with tool/function calling."""
        mock_session = mock_session_class.return_value

        mock_tool_call = Mock()
        mock_tool_call.id = "call_123"
        mock_tool_call.type = "function"
        mock_tool_call.function = Mock()
        mock_tool_call.function.name = "get_weather"
        mock_tool_call.function.arguments = '{"location": "Paris"}'

        mock_response = Mock()
        mock_response.text = "The weather in Paris is 72°F and sunny."
        mock_response.tool_calls = [mock_tool_call]

        mock_session.generate.return_value = mock_response
        mock_get_session_func.return_value = mock_session_class

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g., San Francisco, CA",
                            }
                        },
                        "required": ["location"],
                    },
                },
            }
        ]

        def get_weather(location: str) -> str:
            return f"Weather in {location}"

        response = completion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "What's the weather in Paris?"}],
            tools=tools,
            tool_functions={"get_weather": get_weather},
        )

        assert response is not None
        assert response.choices[0].message.tool_calls is not None
        assert len(response.choices[0].message.tool_calls) == 1
        tool_call = response.choices[0].message.tool_calls[0]
        assert tool_call.function.name == "get_weather"
        assert tool_call.id == "call_123"
        assert '"location": "Paris"' in tool_call.function.arguments

    @patch(
        "litellm.llms.apple_foundation_models.chat.transformation.get_apple_session_class"
    )
    def test_completion_with_structured_output(
        self, mock_get_session_func, mock_session_class
    ):
        """Test completion with structured output (JSON schema)."""
        mock_session = mock_session_class.return_value

        mock_response = Mock()
        mock_response.parsed = {"name": "John", "age": 30}
        mock_response.tool_calls = None
        mock_session.generate.return_value = mock_response

        mock_get_session_func.return_value = mock_session_class

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "number"},
            },
            "required": ["name", "age"],
        }

        response = completion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "Generate a person"}],
            response_format={"type": "json_schema", "json_schema": {"schema": schema}},
        )

        assert response is not None
        call_kwargs = mock_session.generate.call_args[1]
        assert "schema" in call_kwargs

    @patch(
        "litellm.llms.apple_foundation_models.chat.transformation.get_apple_session_class"
    )
    def test_completion_with_pydantic_model(
        self, mock_get_session_func, mock_session_class
    ):
        """Test completion with Pydantic model for structured output."""

        class Person(BaseModel):
            name: str
            age: int
            city: str

        mock_session = mock_session_class.return_value
        mock_response = Mock()
        mock_response.parsed = {"name": "Alice", "age": 30, "city": "Paris"}
        mock_response.tool_calls = None
        mock_session.generate.return_value = mock_response

        mock_get_session_func.return_value = mock_session_class

        response = completion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "Generate a person"}],
            response_format=Person,
        )

        assert response is not None
        call_kwargs = mock_session.generate.call_args[1]
        assert "schema" in call_kwargs

        schema = call_kwargs["schema"]
        assert "properties" in schema
        assert "name" in schema["properties"]
        assert "age" in schema["properties"]
        assert "city" in schema["properties"]

    def test_model_prefix_detection(self):
        """Test that model prefix is properly detected."""
        from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider

        model, provider, api_key, api_base = get_llm_provider(
            model="apple_foundation_models/system"
        )

        assert provider == "apple_foundation_models"
        assert model == "system"

    @patch(
        "litellm.llms.apple_foundation_models.chat.transformation.get_apple_session_class"
    )
    def test_usage_tracking(self, mock_get_session_func, mock_session_class):
        """Test that token usage is tracked (estimated)."""
        mock_get_session_func.return_value = mock_session_class

        response = completion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "Hello"}],
        )

        assert response.usage is not None
        assert response.usage.prompt_tokens > 0
        assert response.usage.completion_tokens > 0
        assert response.usage.total_tokens > 0


class TestAppleFoundationModelsRealIntegration:
    """Real integration tests against Apple Foundation Models API."""

    @pytest.fixture(autouse=True)
    def check_availability(self):
        """Check if Apple Foundation Models is available before each test."""
        try:
            if not apple_intelligence_available():
                pytest.skip("Apple Intelligence is not available")
            from litellm.llms.apple_foundation_models.common_utils import (
                get_apple_session_class,
            )

            get_apple_session_class()
            yield
        except ImportError as e:
            pytest.skip(f"apple-foundation-models package not installed: {e}")
        except RuntimeError as e:
            pytest.skip(f"Apple Intelligence not available: {e}")
        except Exception as e:
            pytest.skip(f"Cannot connect to Apple Foundation Models: {e}")

    def test_real_basic_completion(self):
        """Test real completion with Apple Foundation Models."""
        response = completion(
            model="apple_foundation_models/system",
            messages=[
                {"role": "user", "content": "Say 'Hello, World!' and nothing else."}
            ],
            max_tokens=50,
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert response.choices[0].message.role == "assistant"
        assert response.usage is not None
        assert response.usage.total_tokens > 0

        print(f"\n✅ Response: {response.choices[0].message.content}")

    def test_real_completion_with_system_message(self):
        """Test completion with system message."""
        response = completion(
            model="apple_foundation_models/system",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that speaks like a pirate.",
                },
                {"role": "user", "content": "Introduce yourself in one sentence."},
            ],
            max_tokens=100,
        )

        assert response.choices[0].message.content
        print(f"\n✅ Pirate response: {response.choices[0].message.content}")

    def test_real_completion_with_temperature(self):
        """Test that temperature parameter is accepted."""
        response = completion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "Count to 3."}],
            temperature=0.1,
            max_tokens=50,
        )

        assert response.choices[0].message.content
        print(f"\n✅ Low temp response: {response.choices[0].message.content}")

    def test_real_completion_with_max_tokens(self):
        """Test max_tokens parameter."""
        response = completion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "Write a very short poem about AI."}],
            max_tokens=50,
        )

        assert response.choices[0].message.content
        assert len(response.choices[0].message.content) < 500
        print(f"\n✅ Short response: {response.choices[0].message.content}")

    def test_real_sync_streaming(self):
        """Test real sync streaming completion."""
        response = completion(
            model="apple_foundation_models/system",
            messages=[
                {"role": "user", "content": "Count from 1 to 5, one number per line."}
            ],
            stream=True,
            max_tokens=100,
        )

        chunks = []
        for chunk in response:
            if (
                hasattr(chunk, "choices")
                and chunk.choices
                and chunk.choices[0].delta.content
            ):
                chunks.append(chunk.choices[0].delta.content)

        full_response = "".join(chunks)
        assert chunks, "Should receive at least one chunk"
        assert full_response, "Should have some content"
        print(f"\n✅ Full streamed response (sync): {full_response}")

    @pytest.mark.asyncio
    async def test_real_async_streaming(self):
        """Test real async streaming completion."""
        response = await acompletion(
            model="apple_foundation_models/system",
            messages=[
                {"role": "user", "content": "Count from 1 to 5, one number per line."}
            ],
            stream=True,
            max_tokens=100,
        )

        chunks = []
        async for chunk in response:
            if (
                hasattr(chunk, "choices")
                and chunk.choices
                and chunk.choices[0].delta.content
            ):
                chunks.append(chunk.choices[0].delta.content)

        full_response = "".join(chunks)
        assert chunks, "Should receive at least one chunk"
        assert full_response, "Should have some content"
        print(f"\n✅ Full streamed response (async): {full_response}")

    def test_real_conversation_context(self):
        """Test that multi-turn conversation messages are accepted."""
        response = completion(
            model="apple_foundation_models/system",
            messages=[
                {"role": "user", "content": "My name is Alice."},
                {"role": "assistant", "content": "Hello Alice!"},
                {"role": "user", "content": "Say hello back."},
            ],
            max_tokens=50,
        )

        content = response.choices[0].message.content
        assert len(content) > 0
        print(f"\n✅ Multi-turn response: {content}")

    def test_real_error_handling_empty_message(self):
        """Test that empty messages are handled gracefully."""
        response = completion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": ""}],
            max_tokens=50,
        )

        assert response.choices[0].message.content
        print(f"\n✅ Empty message response: {response.choices[0].message.content}")

    def test_real_multiple_requests(self):
        """Test that multiple sequential requests work correctly."""
        for i in range(3):
            response = completion(
                model="apple_foundation_models/system",
                messages=[{"role": "user", "content": f"Say the number {i+1}."}],
                max_tokens=20,
            )
            assert response.choices[0].message.content
            print(f"\n✅ Request {i+1}: {response.choices[0].message.content}")

    def test_real_longer_response(self):
        """Test with a request that generates a longer response."""
        response = completion(
            model="apple_foundation_models/system",
            messages=[
                {
                    "role": "user",
                    "content": "Write a short 3-sentence story about a robot.",
                }
            ],
            max_tokens=200,
        )

        content = response.choices[0].message.content
        assert len(content) > 50
        print(f"\n✅ Story: {content}")

    def test_real_special_characters(self):
        """Test handling of special characters."""
        response = completion(
            model="apple_foundation_models/system",
            messages=[
                {
                    "role": "user",
                    "content": "Echo this: Hello! 👋 #AI @2024 $100 50% <test>",
                }
            ],
            max_tokens=100,
        )

        assert response.choices[0].message.content
        print(f"\n✅ Special chars response: {response.choices[0].message.content}")

    @pytest.mark.asyncio
    async def test_real_async_non_streaming(self):
        """Test async non-streaming completion."""
        response = await acompletion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "Say 'Hello' and nothing else."}],
            stream=False,
            max_tokens=50,
        )

        assert response is not None
        assert hasattr(response, "choices")
        assert len(response.choices) > 0
        assert response.choices[0].message.content
        assert response.usage is not None
        assert response.usage.total_tokens > 0

        print(
            f"\n✅ Async non-streaming response: {response.choices[0].message.content}"
        )

    @pytest.mark.asyncio
    async def test_real_async_structured_output(self):
        """Test async structured output with JSON schema."""
        import json

        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "reasoning": {"type": "string"},
            },
            "required": ["answer"],
        }

        response = await acompletion(
            model="apple_foundation_models/system",
            messages=[
                {
                    "role": "user",
                    "content": "What is 2+2? Answer with JSON containing 'answer' and 'reasoning'.",
                }
            ],
            response_format={"type": "json_schema", "json_schema": {"schema": schema}},
            max_tokens=100,
        )

        content = response.choices[0].message.content
        print(f"\n✅ Async structured output: {content}")

        data = json.loads(content)
        assert isinstance(data, dict)
        assert "answer" in data
        print(f"✅ Parsed async JSON successfully: {data}")


class TestAppleFoundationModelsAdvancedFeatures:
    """Test advanced features like tools and structured output."""

    @pytest.fixture(autouse=True)
    def check_availability(self):
        """Check if Apple Foundation Models is available."""
        try:
            if not apple_intelligence_available():
                pytest.skip("Apple Intelligence is not available")
            from litellm.llms.apple_foundation_models.common_utils import (
                get_apple_session_class,
            )

            get_apple_session_class()
            yield
        except (ImportError, RuntimeError) as e:
            pytest.skip(f"Apple Foundation Models not available: {e}")

    def test_real_tool_calling(self):
        """Test real tool/function calling with a simple tool."""

        def get_current_time() -> str:
            """Return a simple timestamp string."""
            import datetime

            return datetime.datetime.now().strftime("%H:%M")

        def calculate_sum(a: float, b: float) -> float:
            """Sum two numbers."""
            return a + b

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "Get the current time",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate_sum",
                    "description": "Calculate the sum of two numbers",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "number", "description": "First number"},
                            "b": {"type": "number", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            },
        ]

        response = completion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "What is 5 plus 7?"}],
            tools=tools,
            tool_functions={
                "get_current_time": get_current_time,
                "calculate_sum": calculate_sum,
            },
            max_tokens=150,
        )

        content = response.choices[0].message.content
        assert content
        print(f"\n✅ Tool calling response: {content}")

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            print(f"✅ Tool calls detected: {len(tool_calls)} tool(s) called")
            for tool_call in tool_calls:
                print(f"  - {tool_call.function.name}({tool_call.function.arguments})")
                assert tool_call.id
                assert tool_call.function.name in ["calculate_sum", "get_current_time"]
        else:
            print("⚠️  No tool calls detected (model may have answered directly)")

        assert (
            "12" in content or "seven" in content.lower() or "tool" in content.lower()
        )

    def test_real_tool_calling_functions_only(self):
        """Test tool calling with only tool_functions (no tools schemas)."""

        def get_weather(location: str, units: str = "celsius") -> str:
            """Get the current weather for a location."""
            return f"Weather in {location}: 22°{units[0].upper()}, sunny"

        def calculate(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        response = completion(
            model="apple_foundation_models/system",
            messages=[
                {
                    "role": "user",
                    "content": "What's the weather in Paris and what's 5 plus 7?",
                }
            ],
            tool_functions=[get_weather, calculate],
            max_tokens=200,
        )

        content = response.choices[0].message.content
        assert content
        print(f"\n✅ Tool calling (functions only) response: {content}")

        assert "paris" in content.lower() or "weather" in content.lower()
        assert "12" in content or "seven" in content.lower()

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            print(f"✅ Tool calls detected: {len(tool_calls)} tool(s) called")
            for tool_call in tool_calls:
                print(f"  - {tool_call.function.name}({tool_call.function.arguments})")
        else:
            print("⚠️  No tool calls detected (model may have answered directly)")

    @pytest.mark.asyncio
    async def test_real_async_tool_calling(self):
        """Test async tool calling with actual function implementations."""

        def add_numbers(a: int, b: int) -> int:
            """Add two numbers together."""
            return a + b

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "add_numbers",
                    "description": "Add two numbers together",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer", "description": "First number"},
                            "b": {"type": "integer", "description": "Second number"},
                        },
                        "required": ["a", "b"],
                    },
                },
            }
        ]

        response = await acompletion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "Add 42 and 58 using tool calling."}],
            tools=tools,
            tool_functions={"add_numbers": add_numbers},
            max_tokens=150,
        )

        content = response.choices[0].message.content
        assert content
        print(f"\n✅ Async tool calling response: {content}")

        assert "100" in content or "one hundred" in content.lower()

        tool_calls = response.choices[0].message.tool_calls
        if tool_calls:
            print(f"✅ Tool calls detected: {len(tool_calls)} tool(s) called")
            for tool_call in tool_calls:
                print(f"  - {tool_call.function.name}({tool_call.function.arguments})")

    def test_real_structured_output(self):
        """Test real structured output with JSON schema."""
        import json

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "city": {"type": "string"},
            },
            "required": ["name", "age"],
        }

        response = completion(
            model="apple_foundation_models/system",
            messages=[
                {
                    "role": "user",
                    "content": "Generate information about Alice who is 30 and lives in Paris. Return only JSON.",
                }
            ],
            response_format={"type": "json_schema", "json_schema": {"schema": schema}},
            max_tokens=150,
        )

        content = response.choices[0].message.content
        print(f"\n✅ Structured output: {content}")

        data = json.loads(content)
        assert isinstance(data, dict)
        assert "name" in data and "age" in data
        assert isinstance(data["age"], int)

    def test_real_structured_output_simple(self):
        """Test structured output with a simpler schema."""
        import json

        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
            },
            "required": ["answer"],
        }

        response = completion(
            model="apple_foundation_models/system",
            messages=[
                {
                    "role": "user",
                    "content": "Is the sky blue? Answer with JSON containing 'answer' (yes/no) and 'confidence' (0-1).",
                }
            ],
            response_format={"type": "json_schema", "json_schema": {"schema": schema}},
            max_tokens=100,
        )

        content = response.choices[0].message.content
        print(f"\n✅ Simple structured output: {content}")

        data = json.loads(content)
        assert "answer" in data

    def test_real_pydantic_structured_output(self):
        """Test structured output with Pydantic model (as documented)."""
        import json

        class Person(BaseModel):
            name: str
            age: int
            city: str

        response = completion(
            model="apple_foundation_models/system",
            messages=[
                {
                    "role": "user",
                    "content": "Extract person info: Alice is 30 and lives in Paris.",
                }
            ],
            response_format=Person,  # Pass Pydantic model directly
            max_tokens=150,
        )

        content = response.choices[0].message.content
        print(f"\n✅ Pydantic structured output: {content}")

        # Response is automatically formatted as JSON
        data = json.loads(content)
        assert isinstance(data, dict)
        assert "name" in data and "age" in data and "city" in data
        assert isinstance(data["age"], int)
        print(f"✅ Parsed Pydantic response: {data}")

    def test_real_multiple_sequential_requests(self):
        """Test that multiple sequential requests work correctly."""
        response1 = completion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "Say the number 1."}],
            max_tokens=10,
        )
        assert response1.choices[0].message.content
        print(f"\n✅ First request response: {response1.choices[0].message.content}")

        response2 = completion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "Say the number 2."}],
            max_tokens=10,
        )
        assert response2.choices[0].message.content
        print(f"✅ Second request response: {response2.choices[0].message.content}")

        response3 = completion(
            model="apple_foundation_models/system",
            messages=[{"role": "user", "content": "Say the number 3."}],
            max_tokens=10,
        )
        assert response3.choices[0].message.content
        print(f"✅ Third request response: {response3.choices[0].message.content}")
        print("✅ Multiple sequential requests verified")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
