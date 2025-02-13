from base_llm_unit_tests import BaseLLMChatTest
import pytest
import sys
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import completion, ModelResponse, RateLimitError


class TestBedrockTestSuite:
    def get_base_completion_call_args(self) -> dict:
        litellm._turn_on_debug()
        return {
            "model": "bedrock/invoke/us.meta.llama3-3-70b-instruct-v1:0",
        }

    def test_bedrock_llama_tool_calling(self):
        try:
            litellm.set_verbose = True
            litellm._turn_on_debug()
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
            messages = [
                {
                    "role": "user",
                    "content": "What's the weather like in Boston today in fahrenheit?",
                }
            ]
            request_args = {
                "messages": messages,
                "tools": tools,
            }
            request_args.update(self.get_base_completion_call_args())
            response: ModelResponse = completion(**request_args)  # type: ignore
            print(f"response: {response}")
            # Add any assertions here to check the response
            assert isinstance(
                response.choices[0].message.tool_calls[0].function.name, str
            )
            assert isinstance(
                response.choices[0].message.tool_calls[0].function.arguments, str
            )
            messages.append(
                response.choices[0].message.model_dump()
            )  # Add assistant tool invokes
            tool_result = (
                '{"location": "Boston", "temperature": "72", "unit": "fahrenheit"}'
            )
            # Add user submitted tool results in the OpenAI format
            messages.append(
                {
                    "tool_call_id": response.choices[0].message.tool_calls[0].id,
                    "role": "tool",
                    "name": response.choices[0].message.tool_calls[0].function.name,
                    "content": tool_result,
                }
            )
            # In the second response, Claude should deduce answer from tool results
            request_2_args = {
                "messages": messages,
                "tools": tools,
            }
            request_2_args.update(self.get_base_completion_call_args())
            second_response: ModelResponse = completion(**request_2_args)  # type: ignore
            print(f"second response: {second_response}")
            assert isinstance(second_response.choices[0].message.content, str)
        except RateLimitError:
            pass
        except Exception as e:
            pytest.fail(f"Error occurred: {e}")
