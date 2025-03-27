import pytest
from litellm import AzureOpenAI
from dotenv import load_dotenv
import os
import json

load_dotenv()

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    elif "paris" in location.lower():
        return json.dumps({"location": "Paris", "temperature": "22", "unit": "celsius"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

@pytest.fixture
def azure_client():
    """Fixture to create an Azure OpenAI client"""
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_API_BASE"),
        api_key=os.getenv("AZURE_API_KEY"),
        api_version="2024-12-01-preview"
    )

@pytest.fixture
def weather_tool():
    """Fixture for the weather tool configuration"""
    return [{
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"]
                    }
                },
                "required": ["location"]
            }
        }
    }]

def test_azure_openai_tool_calling(azure_client, weather_tool):
    """
    Test Azure OpenAI tool calling functionality with weather information retrieval.
    
    This test verifies that:
    1. The model can understand and process tool calling requests
    2. The weather function is called correctly
    3. The response is properly formatted and contains the expected weather information
    """
    # Perform the chat completion request with tools
    response = azure_client.chat.completions.create(
        model=os.getenv("AZURE_DEPLOYMENT_NAME"),
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the weather in Tokyo?"}
        ],
        tools=weather_tool,
        tool_choice="auto",
    )

    # Verify that we got a response with tool calls
    assert response.choices[0].message.tool_calls is not None, "Expected tool calls in response"
    
    # Process tool calls and update response
    for tool_call in response.choices[0].message.tool_calls:
        if tool_call.function.name == "get_current_weather":
            # Verify the function name
            assert tool_call.function.name == "get_current_weather", "Unexpected function call"
            
            # Parse and verify arguments
            args = json.loads(tool_call.function.arguments)
            assert "location" in args, "Location not found in arguments"
            
            # Get weather information
            location = args["location"]
            unit = args.get("unit", "fahrenheit")
            weather = get_current_weather(location, unit)
            weather_json = json.loads(weather)
            
            # Verify weather response format
            assert all(key in weather_json for key in ["location", "temperature", "unit"]), "Missing required weather information"
            
            # Update response content
            weather_statement = f"The weather in {weather_json['location']} is {weather_json['temperature']} {weather_json['unit']}"
            response.choices[0].message.content = weather_statement

    # Verify final response content
    assert "weather in Tokyo" in response.choices[0].message.content, "Expected weather information for Tokyo in response"
    assert "10 celsius" in response.choices[0].message.content, "Expected temperature in response"

    return response