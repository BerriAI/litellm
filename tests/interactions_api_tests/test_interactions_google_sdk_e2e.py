#!/usr/bin/env python3
"""
E2E Tests for Interactions API using Google GenAI SDK.

Tests the Interactions -> Responses API bridge by using the actual Google GenAI SDK
client to call a LiteLLM proxy endpoint, which routes to Anthropic.

This validates the full flow:
    Google SDK Client -> LiteLLM Proxy -> Interactions Bridge -> Responses API -> Anthropic

Run with:
    ANTHROPIC_API_KEY=<key> python -m pytest test_interactions_google_sdk_e2e.py -v -s
"""

import asyncio
import json
import os
import subprocess
import sys
import time
import warnings

import pytest

# Suppress experimental warnings
warnings.filterwarnings("ignore", message=".*Interactions usage is experimental.*")
warnings.filterwarnings("ignore", message=".*Async interactions client cannot use aiohttp.*")

# Configuration
PROXY_PORT = 4001
PROXY_HOST = "localhost"
PROXY_BASE_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"
PROXY_API_KEY = "sk-test-1234"
MODEL = "anthropic/claude-3-5-haiku-20241022"


@pytest.fixture(scope="module")
def proxy_server():
    """Start a mini proxy server for testing."""
    # Get Anthropic API key
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        pytest.skip("ANTHROPIC_API_KEY not set")
    
    # Get workspace path
    workspace_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Start mini proxy server
    proxy_code = f'''
import os
import sys
import json

os.environ["ANTHROPIC_API_KEY"] = "{anthropic_key}"
sys.path.insert(0, "{workspace_path}")

from fastapi import FastAPI, Request, Header
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn

app = FastAPI()

@app.post("/v1beta/interactions")
@app.post("/interactions")
async def create_interaction(
    request: Request,
    authorization: str = Header(None),
    x_goog_api_key: str = Header(None),
):
    from litellm import interactions
    
    api_key = None
    if authorization:
        api_key = authorization.replace("Bearer ", "")
    elif x_goog_api_key:
        api_key = x_goog_api_key
    
    if api_key != "{PROXY_API_KEY}":
        return JSONResponse({{"error": "Unauthorized"}}, status_code=401)
    
    data = await request.json()
    
    model = data.get("model", "{MODEL}")
    input_data = data.get("input")
    stream = data.get("stream", False)
    system_instruction = data.get("system_instruction")
    tools = data.get("tools")
    generation_config = data.get("generation_config", {{}})
    
    try:
        kwargs = {{
            "model": model,
            "input": input_data,
            "api_key": os.environ["ANTHROPIC_API_KEY"],
        }}
        
        if system_instruction:
            kwargs["system_instruction"] = system_instruction
        if tools:
            kwargs["tools"] = tools
        if generation_config.get("temperature"):
            kwargs["temperature"] = generation_config["temperature"]
        if generation_config.get("max_output_tokens"):
            kwargs["max_output_tokens"] = generation_config["max_output_tokens"]
        
        if stream:
            kwargs["stream"] = True
            response = interactions.create(**kwargs)
            
            async def generate():
                for chunk in response:
                    chunk_data = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
                    yield f"data: {{json.dumps(chunk_data)}}\\n\\n"
                yield "data: [DONE]\\n\\n"
            
            return StreamingResponse(generate(), media_type="text/event-stream")
        else:
            response = interactions.create(**kwargs)
            response_data = response.model_dump() if hasattr(response, "model_dump") else dict(response)
            return JSONResponse(response_data)
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({{"error": str(e)}}, status_code=500)

@app.get("/health")
async def health():
    return {{"status": "ok"}}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={PROXY_PORT}, log_level="warning")
'''
    
    # Write proxy script
    proxy_script = "/tmp/test_interactions_proxy.py"
    with open(proxy_script, "w") as f:
        f.write(proxy_code)
    
    # Start proxy in background
    proc = subprocess.Popen(
        [sys.executable, proxy_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    
    # Wait for proxy to be ready
    max_retries = 30
    for i in range(max_retries):
        try:
            import httpx
            response = httpx.get(f"{PROXY_BASE_URL}/health", timeout=2)
            if response.status_code == 200:
                break
        except Exception:
            pass
        time.sleep(1)
    else:
        proc.terminate()
        pytest.fail("Proxy server failed to start")
    
    yield proc
    
    # Cleanup
    proc.terminate()
    proc.wait(timeout=5)


class TestInteractionsAPIWithGoogleSDK:
    """E2E tests using Google GenAI SDK to call Interactions API."""
    
    def test_simple_interaction(self, proxy_server):
        """Test simple text interaction."""
        from google import genai
        from google.genai.types import HttpOptions
        
        client = genai.Client(
            api_key=PROXY_API_KEY,
            http_options=HttpOptions(base_url=PROXY_BASE_URL, api_version="v1beta")
        )
        
        response = client.interactions.create(
            model=MODEL,
            input="What is 2 + 2? Answer with just the number."
        )
        
        assert response is not None
        assert str(response.status).lower() == "completed"
        assert response.outputs is not None
        assert len(response.outputs) > 0
        
        output_text = str(response.outputs)
        assert "4" in output_text, f"Expected '4' in output, got: {output_text}"
    
    def test_streaming_interaction(self, proxy_server):
        """Test streaming interaction."""
        from google import genai
        from google.genai.types import HttpOptions
        
        client = genai.Client(
            api_key=PROXY_API_KEY,
            http_options=HttpOptions(base_url=PROXY_BASE_URL, api_version="v1beta")
        )
        
        response_stream = client.interactions.create(
            model=MODEL,
            input="Count from 1 to 3. Just numbers, comma separated.",
            stream=True,
        )
        
        chunks = []
        for chunk in response_stream:
            chunks.append(chunk)
        
        assert len(chunks) > 0, "Expected streaming chunks"
        
        # Verify event types
        event_types = [str(c.event_type) for c in chunks if hasattr(c, 'event_type')]
        assert "interaction.start" in event_types, "Expected interaction.start event"
        assert "interaction.complete" in event_types, "Expected interaction.complete event"
    
    def test_system_instruction(self, proxy_server):
        """Test with system instruction."""
        from google import genai
        from google.genai.types import HttpOptions
        
        client = genai.Client(
            api_key=PROXY_API_KEY,
            http_options=HttpOptions(base_url=PROXY_BASE_URL, api_version="v1beta")
        )
        
        response = client.interactions.create(
            model=MODEL,
            input="What are you?",
            system_instruction="You are a robot. Always start with 'Beep boop!'",
        )
        
        assert response is not None
        assert response.outputs is not None
        
        output_text = str(response.outputs).lower()
        assert "beep" in output_text or "boop" in output_text, f"Expected robot response, got: {output_text}"
    
    def test_multi_turn_conversation(self, proxy_server):
        """Test multi-turn conversation."""
        from google import genai
        from google.genai.types import HttpOptions
        
        client = genai.Client(
            api_key=PROXY_API_KEY,
            http_options=HttpOptions(base_url=PROXY_BASE_URL, api_version="v1beta")
        )
        
        response = client.interactions.create(
            model=MODEL,
            input=[
                {"role": "user", "content": [{"type": "text", "text": "My name is Alice."}]},
                {"role": "model", "content": [{"type": "text", "text": "Hello Alice!"}]},
                {"role": "user", "content": [{"type": "text", "text": "What is my name?"}]},
            ]
        )
        
        assert response is not None
        assert response.outputs is not None
        
        output_text = str(response.outputs).lower()
        assert "alice" in output_text, f"Expected 'Alice' in response, got: {output_text}"
    
    def test_function_calling(self, proxy_server):
        """Test function/tool calling."""
        from google import genai
        from google.genai.types import HttpOptions
        
        client = genai.Client(
            api_key=PROXY_API_KEY,
            http_options=HttpOptions(base_url=PROXY_BASE_URL, api_version="v1beta")
        )
        
        response = client.interactions.create(
            model=MODEL,
            input="What's the weather in Tokyo?",
            tools=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {"type": "string", "description": "City name"}
                        },
                        "required": ["location"]
                    }
                }
            ]
        )
        
        assert response is not None
        assert response.outputs is not None
        
        # Check for function call
        output_str = str(response.outputs)
        has_function_call = "function_call" in output_str or "get_weather" in output_str
        assert has_function_call, f"Expected function call in output, got: {output_str}"
    
    @pytest.mark.asyncio
    async def test_async_interaction(self, proxy_server):
        """Test async interaction."""
        from google import genai
        from google.genai.types import HttpOptions
        
        client = genai.Client(
            api_key=PROXY_API_KEY,
            http_options=HttpOptions(base_url=PROXY_BASE_URL, api_version="v1beta")
        )
        
        response = await client.aio.interactions.create(
            model=MODEL,
            input="Say hello."
        )
        
        assert response is not None
        assert response.outputs is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
