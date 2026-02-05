#!/usr/bin/env python3
"""
Manual test: Verify A2A streaming returns text/event-stream (SSE format)

This script:
1. Starts a simple A2A agent on port 10001
2. Starts LiteLLM proxy on port 4000 with the agent registered
3. Makes a streaming request to the proxy's A2A gateway
4. Validates Content-Type header and SSE body format

Run: python test_a2a_sse_manual.py

Expected output:
  ✅ Content-Type: text/event-stream
  ✅ SSE framing: data: {...}\n\n
"""

import asyncio
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

import httpx

AGENT_PORT = 10001
PROXY_PORT = 4000
TIMEOUT = 60


def start_agent():
    """Start a simple A2A agent using LiteLLM's a2a_protocol."""
    agent_code = '''
import asyncio
from litellm.a2a_protocol import A2AServer, BaseA2AAgent

class TestAgent(BaseA2AAgent):
    @property
    def name(self) -> str:
        return "test-agent"
    
    @property
    def description(self) -> str:
        return "Test agent for SSE validation"
    
    @property
    def streaming(self) -> bool:
        return True

    async def invoke(self, query, session_id=None):
        return {"role": "agent", "parts": [{"kind": "text", "text": "Hello from test agent!"}]}

    async def invoke_streaming(self, query, session_id=None):
        for word in ["Hello", "from", "streaming", "agent!"]:
            yield {"role": "agent", "parts": [{"kind": "text", "text": word + " "}]}
            await asyncio.sleep(0.1)

async def main():
    server = A2AServer(agent=TestAgent(), host="0.0.0.0", port=10001)
    await server.start()

if __name__ == "__main__":
    asyncio.run(main())
'''
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(agent_code)
        agent_file = f.name
    
    proc = subprocess.Popen(
        [sys.executable, agent_file],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc, agent_file


def start_proxy():
    """Start LiteLLM proxy with A2A agent registered."""
    config = {
        "model_list": [
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "gpt-4o-mini",
                }
            }
        ],
        "a2a_config": {
            "agents": [
                {
                    "agent_id": "test-agent",
                    "api_base": f"http://localhost:{AGENT_PORT}",
                }
            ]
        },
        "general_settings": {
            "master_key": "sk-test-key"
        }
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        import yaml
        yaml.dump(config, f)
        config_file = f.name
    
    proc = subprocess.Popen(
        [sys.executable, "-m", "litellm", "--config", config_file, "--port", str(PROXY_PORT)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return proc, config_file


async def wait_for_server(url: str, timeout: int = 30):
    """Wait for a server to be ready."""
    start = time.time()
    async with httpx.AsyncClient() as client:
        while time.time() - start < timeout:
            try:
                resp = await client.get(url, timeout=2)
                if resp.status_code < 500:
                    return True
            except Exception:
                pass
            await asyncio.sleep(0.5)
    return False


async def test_streaming():
    """Test the A2A streaming endpoint."""
    print("\n" + "=" * 60)
    print("Testing A2A Streaming SSE Format")
    print("=" * 60)
    
    url = f"http://localhost:{PROXY_PORT}/a2a/test-agent"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer sk-test-key",
    }
    payload = {
        "jsonrpc": "2.0",
        "id": "test-123",
        "method": "message/stream",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "Hello"}],
                "messageId": "msg-001",
            }
        }
    }
    
    print(f"\n📡 POST {url}")
    print(f"   Method: message/stream")
    
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload, headers=headers, timeout=30) as resp:
            # Check Content-Type header
            content_type = resp.headers.get("content-type", "")
            print(f"\n📋 Response Headers:")
            print(f"   Content-Type: {content_type}")
            
            if "text/event-stream" in content_type:
                print("   ✅ Correct! Uses text/event-stream (SSE)")
            elif "application/x-ndjson" in content_type:
                print("   ❌ WRONG! Still using application/x-ndjson")
                return False
            else:
                print(f"   ⚠️  Unexpected Content-Type")
            
            # Check body format
            print(f"\n📦 Response Body (SSE events):")
            events = []
            async for line in resp.aiter_lines():
                if line.strip():
                    events.append(line)
                    print(f"   {line[:100]}{'...' if len(line) > 100 else ''}")
            
            # Validate SSE framing
            print(f"\n🔍 SSE Format Validation:")
            all_valid = True
            for i, event in enumerate(events):
                if event.startswith("data: "):
                    try:
                        payload_str = event[6:]  # Remove "data: " prefix
                        json.loads(payload_str)
                        print(f"   Event {i+1}: ✅ Valid SSE (data: <json>)")
                    except json.JSONDecodeError as e:
                        print(f"   Event {i+1}: ❌ Invalid JSON: {e}")
                        all_valid = False
                else:
                    print(f"   Event {i+1}: ❌ Missing 'data: ' prefix")
                    all_valid = False
            
            if all_valid and events:
                print(f"\n✅ ALL TESTS PASSED - A2A streaming uses correct SSE format")
                return True
            else:
                print(f"\n❌ TESTS FAILED")
                return False


async def main():
    agent_proc = None
    proxy_proc = None
    agent_file = None
    config_file = None
    
    try:
        print("🚀 Starting test A2A agent on port", AGENT_PORT)
        agent_proc, agent_file = start_agent()
        
        print("⏳ Waiting for agent to be ready...")
        if not await wait_for_server(f"http://localhost:{AGENT_PORT}/.well-known/agent.json"):
            print("❌ Agent failed to start")
            # Print agent stderr for debugging
            if agent_proc.stderr:
                print("Agent stderr:", agent_proc.stderr.read().decode())
            return 1
        print("✅ Agent ready")
        
        print("\n🚀 Starting LiteLLM proxy on port", PROXY_PORT)
        proxy_proc, config_file = start_proxy()
        
        print("⏳ Waiting for proxy to be ready...")
        if not await wait_for_server(f"http://localhost:{PROXY_PORT}/health"):
            print("❌ Proxy failed to start")
            if proxy_proc.stderr:
                print("Proxy stderr:", proxy_proc.stderr.read().decode())
            return 1
        print("✅ Proxy ready")
        
        # Run the test
        success = await test_streaming()
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Cleanup
        if agent_proc:
            agent_proc.terminate()
            agent_proc.wait()
        if proxy_proc:
            proxy_proc.terminate()
            proxy_proc.wait()
        if agent_file and os.path.exists(agent_file):
            os.unlink(agent_file)
        if config_file and os.path.exists(config_file):
            os.unlink(config_file)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
