#!/usr/bin/env python3
"""
Direct test of AgentCore API (no LiteLLM) to verify thinking/reasoning works.
"""

import json
import requests
import uuid
from urllib.parse import quote

# Configuration
COGNITO_CLIENT_ID = "65su16qe567iap1010l914dhbg"
COGNITO_CLIENT_SECRET = "pa01tlqslkokulsq0mq910v7238oa8g6vins0lqd966rmb8ck19"
COGNITO_TOKEN_URL = "https://apro-chat.auth.eu-central-1.amazoncognito.com/oauth2/token"
AGENTCORE_RUNTIME_ARN = "arn:aws:bedrock-agentcore:eu-west-1:515966504419:runtime/apro_sandbox_tomas_genai_v2_ut_messan_runtime-N9x7Ce3CcD"

def get_token():
    """Get JWT token from Cognito."""
    print("🔐 Getting token...")
    resp = requests.post(
        COGNITO_TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": COGNITO_CLIENT_ID,
            "client_secret": COGNITO_CLIENT_SECRET,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    token = resp.json()["access_token"]
    print(f"✅ Token: {token[:50]}...")
    return token

def test_agentcore(token: str):
    """Test AgentCore directly."""
    # Parse ARN
    region = AGENTCORE_RUNTIME_ARN.split(":")[3]

    # Build URL with URL-encoded ARN
    encoded_arn = quote(AGENTCORE_RUNTIME_ARN, safe="")
    url = f"https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{encoded_arn}/invocations"

    # Generate session ID (must be 33+ chars)
    session_id = f"test-session-{uuid.uuid4()}"

    prompt = "Think step by step: What is 23 * 47?"

    print(f"\n🌐 URL: {url}")
    print(f"📤 Prompt: {prompt}")
    print(f"📡 Session: {session_id}")
    print("\n" + "="*60)
    print("  RAW SSE EVENTS")
    print("="*60 + "\n")

    # Make streaming request
    resp = requests.post(
        url,
        json={"prompt": prompt},
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
        },
        stream=True,
    )

    print(f"Status: {resp.status_code}\n")

    if resp.status_code != 200:
        print(f"❌ Error: {resp.text}")
        return

    # Process events
    reasoning_found = False
    content_parts = []

    for line in resp.iter_lines():
        if line:
            decoded = line.decode('utf-8')
            print(decoded)

            # Check for reasoning
            if 'reasoning' in decoded.lower():
                reasoning_found = True
                print("  👆 REASONING EVENT!")

            # Extract content
            if decoded.startswith("data:"):
                try:
                    data = json.loads(decoded[5:].strip())
                    if "event" in data:
                        evt = data["event"]
                        if "contentBlockDelta" in evt:
                            text = evt["contentBlockDelta"].get("delta", {}).get("text", "")
                            if text:
                                content_parts.append(text)
                except:
                    pass

    print("\n" + "="*60)
    print("  SUMMARY")
    print("="*60)

    if content_parts:
        print(f"\n📄 Response: {''.join(content_parts)}")

    if reasoning_found:
        print("\n✅ REASONING/THINKING DETECTED!")
    else:
        print("\n❌ No reasoning detected - agent needs thinking enabled")

if __name__ == "__main__":
    token = get_token()
    test_agentcore(token)
