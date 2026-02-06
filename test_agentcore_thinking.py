#!/usr/bin/env python3
"""
Test script for Bedrock AgentCore thinking/reasoning feature.

This script tests whether the AgentCore agent emits reasoning events
and whether LiteLLM correctly captures them.

Usage:
    python test_agentcore_thinking.py

Requirements:
    - pip install requests httpx
"""

import os
import json
import requests
import httpx
from typing import Optional

# ============================================================================
# Configuration - Update these values
# ============================================================================

# Cognito OAuth2 credentials
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID", "65su16qe567iap1010l914dhbg")
COGNITO_CLIENT_SECRET = os.getenv("COGNITO_CLIENT_SECRET", "pa01tlqslkokulsq0mq910v7238oa8g6vins0lqd966rmb8ck19")
COGNITO_TOKEN_URL = os.getenv(
    "COGNITO_TOKEN_URL",
    "https://apro-chat.auth.eu-central-1.amazoncognito.com/oauth2/token"
)

# AgentCore Runtime ARN
AGENTCORE_RUNTIME_ARN = os.getenv(
    "AGENTCORE_RUNTIME_ARN",
    "arn:aws:bedrock-agentcore:eu-west-1:515966504419:runtime/apro_sandbox_tomas_genai_v2_ut_messan_runtime-N9x7Ce3CcD"
)

# Enable debug logging
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# ============================================================================
# Helper Functions
# ============================================================================

def get_cognito_token() -> str:
    """Get JWT token from Cognito using client credentials flow."""
    print("🔐 Getting JWT token from Cognito...")

    response = requests.post(
        COGNITO_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": COGNITO_CLIENT_ID,
            "client_secret": COGNITO_CLIENT_SECRET,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )

    if response.status_code != 200:
        raise Exception(f"Failed to get token: {response.status_code} - {response.text}")

    token = response.json()["access_token"]
    print(f"✅ Got token (first 50 chars): {token[:50]}...")
    return token


def print_separator(title: str):
    """Print a visual separator."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}\n")


def parse_arn(arn: str) -> tuple:
    """Parse ARN to extract region and runtime ID."""
    parts = arn.split(":")
    region = parts[3]
    runtime_id = parts[5].split("/")[1]
    return region, runtime_id


def generate_session_id() -> str:
    """Generate a session ID (must be 33+ chars)."""
    import uuid
    return f"session-{uuid.uuid4()}-test"


# ============================================================================
# Test Functions
# ============================================================================

def test_direct_agentcore_streaming(token: str):
    """Test AgentCore streaming by making direct HTTP calls."""
    print_separator("Test: Direct AgentCore Streaming")

    region, runtime_id = parse_arn(AGENTCORE_RUNTIME_ARN)
    session_id = generate_session_id()

    # AgentCore invoke endpoint
    url = f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{runtime_id}/sessions/{session_id}/invoke"

    prompt = "Think step by step: What is 23 * 47? Show your reasoning process."

    print(f"🌐 URL: {url}")
    print(f"📤 Prompt: {prompt}")
    print(f"📡 Raw SSE events:\n")

    # Build request
    payload = {"prompt": prompt}
    request_body = {"payload": json.dumps(payload)}

    try:
        with httpx.Client(timeout=120.0) as client:
            with client.stream(
                "POST",
                url,
                json=request_body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                    "Accept": "text/event-stream",
                },
            ) as response:
                print(f"📡 Response status: {response.status_code}")

                if response.status_code != 200:
                    error_body = response.read().decode()
                    print(f"❌ Error: {error_body[:500]}")
                    return False

                reasoning_events = []
                content_events = []
                all_content = []
                event_count = 0

                for line in response.iter_lines():
                    if line:
                        event_count += 1

                        # Print truncated line for readability
                        display_line = line[:120] + "..." if len(line) > 120 else line
                        print(f"  [{event_count}] {display_line}")

                        # Try to parse JSON from "data:" prefix
                        if line.startswith("data:"):
                            try:
                                json_str = line[5:].strip()
                                if json_str:
                                    event_data = json.loads(json_str)

                                    # Check for reasoning events
                                    if event_data.get("reasoning"):
                                        reasoning_events.append(event_data)
                                        text = event_data.get("reasoningText", "")
                                        print(f"      🧠 REASONING: {text[:80]}...")

                                    # Check for content delta
                                    if "event" in event_data:
                                        evt = event_data["event"]
                                        if "contentBlockDelta" in evt:
                                            delta = evt["contentBlockDelta"].get("delta", {})
                                            text = delta.get("text", "")
                                            if text:
                                                all_content.append(text)
                                            # Check for reasoning in delta
                                            if delta.get("reasoningText"):
                                                reasoning_events.append(event_data)
                                                print(f"      🧠 REASONING (delta): {delta['reasoningText'][:80]}...")

                                        content_events.append(event_data)
                            except json.JSONDecodeError:
                                pass

                        # Also check raw line for reasoning keywords
                        elif 'reasoning' in line.lower():
                            reasoning_events.append(line)

                print("\n")
                print_separator("Results Summary")
                print(f"📊 Total events: {event_count}")
                print(f"📝 Content events: {len(content_events)}")
                print(f"🧠 Reasoning events: {len(reasoning_events)}")

                if all_content:
                    full_content = "".join(all_content)
                    print(f"\n📄 Full Response:\n{full_content[:500]}...")

                if reasoning_events:
                    print("\n✅ REASONING DETECTED!")
                    for i, evt in enumerate(reasoning_events[:3]):
                        if isinstance(evt, dict):
                            print(f"   [{i+1}] {json.dumps(evt)[:150]}...")
                        else:
                            print(f"   [{i+1}] {evt[:150]}...")
                    return True
                else:
                    print("\n❌ No reasoning events detected in stream")
                    return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_direct_agentcore_non_streaming(token: str):
    """Test AgentCore non-streaming by making direct HTTP calls."""
    print_separator("Test: Direct AgentCore Non-Streaming")

    region, runtime_id = parse_arn(AGENTCORE_RUNTIME_ARN)
    session_id = generate_session_id()

    # AgentCore invoke endpoint
    url = f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{runtime_id}/sessions/{session_id}/invoke"

    prompt = "What is the capital of France? Answer briefly."

    print(f"🌐 URL: {url}")
    print(f"📤 Prompt: {prompt}")
    print(f"⏳ Waiting for response...\n")

    # Build request
    payload = {"prompt": prompt}
    request_body = {"payload": json.dumps(payload)}

    try:
        response = httpx.post(
            url,
            json=request_body,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            timeout=120.0,
        )

        print(f"📡 Response status: {response.status_code}")

        if response.status_code != 200:
            print(f"❌ Error: {response.text[:500]}")
            return False

        # Print raw response
        raw_text = response.text
        print(f"\n📥 Raw response:\n{raw_text[:1000]}...")

        # Check for reasoning in response
        has_reasoning = 'reasoning' in raw_text.lower()

        if has_reasoning:
            print("\n✅ REASONING DETECTED in response!")
            return True
        else:
            print("\n❌ No reasoning detected in response")
            return False

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_with_litellm_config(token: str):
    """Test using LiteLLM's AmazonAgentCoreConfig."""
    print_separator("Test: LiteLLM AmazonAgentCoreConfig")

    try:
        from litellm.llms.bedrock.chat.agentcore.transformation import AmazonAgentCoreConfig
        from litellm.litellm_core_utils.litellm_logging import Logging
    except ImportError as e:
        print(f"❌ Could not import LiteLLM AgentCore module: {e}")
        return False

    config = AmazonAgentCoreConfig()

    prompt = "What is 2+2?"
    messages = [{"role": "user", "content": prompt}]

    # The model format that LiteLLM expects
    model_str = f"agentcore/{AGENTCORE_RUNTIME_ARN}"

    print(f"📤 Prompt: {prompt}")
    print(f"📦 Model: {model_str}")

    try:
        # Validate environment and get headers
        headers = config.validate_environment(
            headers={},
            model=model_str,
            messages=messages,
            optional_params={},
            litellm_params={},
            api_key=token,
        )

        # Get the complete URL
        url = config.get_complete_url(
            api_base=None,
            api_key=token,
            model=model_str,
            optional_params={},
            litellm_params={},
            stream=True,
        )

        # Transform request
        data = config.transform_request(
            model=model_str,
            messages=messages,
            optional_params={},
            litellm_params={},
            headers=headers,
        )

        print(f"🌐 URL: {url}")
        print(f"📦 Request: {json.dumps(data, indent=2)}")
        print(f"📡 Making streaming request...\n")

        # Make request
        with httpx.Client(timeout=120.0) as client:
            with client.stream("POST", url, json=data, headers=headers) as response:
                print(f"📡 Response status: {response.status_code}")

                if response.status_code != 200:
                    error_body = response.read().decode()
                    print(f"❌ Error: {error_body[:500]}")
                    return False

                reasoning_found = False
                event_count = 0
                all_content = []

                for line in response.iter_lines():
                    if line:
                        event_count += 1
                        display = line[:100] + "..." if len(line) > 100 else line
                        print(f"  [{event_count}] {display}")

                        # Check for reasoning
                        if 'reasoning' in line.lower():
                            reasoning_found = True
                            print(f"      🧠 REASONING DETECTED!")

                        # Try to extract content
                        if line.startswith("data:"):
                            try:
                                event_data = json.loads(line[5:].strip())
                                if "event" in event_data:
                                    evt = event_data["event"]
                                    if "contentBlockDelta" in evt:
                                        delta = evt["contentBlockDelta"].get("delta", {})
                                        text = delta.get("text", "")
                                        if text:
                                            all_content.append(text)
                            except:
                                pass

                print(f"\n📊 Total events: {event_count}")
                if all_content:
                    print(f"📄 Response: {''.join(all_content)[:300]}...")

                if reasoning_found:
                    print("\n✅ REASONING DETECTED!")
                else:
                    print("\n❌ No reasoning detected")

                return reasoning_found

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Main
# ============================================================================

def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║     Bedrock AgentCore Thinking/Reasoning Test Script         ║
╠══════════════════════════════════════════════════════════════╣
║  This script tests whether reasoning events are emitted      ║
║  by your AgentCore agent and captured by LiteLLM.            ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Get authentication token
    try:
        token = get_cognito_token()
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        return

    results = {}

    # Run tests
    results['direct_streaming'] = test_direct_agentcore_streaming(token)
    results['direct_non_streaming'] = test_direct_agentcore_non_streaming(token)
    results['litellm_config'] = test_with_litellm_config(token)

    # Final summary
    print_separator("FINAL SUMMARY")

    for test_name, passed in results.items():
        status = "✅ THINKING FOUND" if passed else "❌ NO THINKING"
        print(f"  {test_name}: {status}")

    if not any(results.values()):
        print("""
╔══════════════════════════════════════════════════════════════╗
║  ⚠️  NO THINKING/REASONING DETECTED                          ║
╠══════════════════════════════════════════════════════════════╣
║  Your AgentCore agent is NOT emitting reasoning events.      ║
║                                                              ║
║  To enable thinking in your Strands agent, configure it:     ║
║                                                              ║
║    bedrock_model = BedrockModel(                             ║
║        model_id="anthropic.claude-sonnet-4-...",             ║
║        additional_request_fields={                           ║
║            "thinking": {                                     ║
║                "type": "enabled",                            ║
║                "budget_tokens": 4096                         ║
║            }                                                 ║
║        }                                                     ║
║    )                                                         ║
╚══════════════════════════════════════════════════════════════╝
""")
    else:
        print("""
╔══════════════════════════════════════════════════════════════╗
║  ✅ THINKING/REASONING DETECTED!                             ║
╠══════════════════════════════════════════════════════════════╣
║  Your AgentCore agent is emitting reasoning events and       ║
║  LiteLLM is correctly capturing them.                        ║
╚══════════════════════════════════════════════════════════════╝
""")


if __name__ == "__main__":
    main()
