#!/usr/bin/env python3
"""
Test script to validate the AgentCore provider implementation
without requiring a deployed agent.
"""

import sys
import os
import json

# Add the parent directory to sys.path to import our AgentCore provider
sys.path.insert(0, os.path.dirname(__file__))

import litellm
from litellm.llms.bedrock.agentcore import AgentCoreConfig

def test_provider_registration():
    """Test that AgentCore provider is properly registered with LiteLLM"""
    print("🔍 Testing AgentCore Provider Registration")
    print("=" * 50)

    # Check if agentcore is in the supported providers
    from litellm.types.utils import LlmProviders

    if hasattr(LlmProviders, 'AGENTCORE'):
        print("✅ AGENTCORE found in LlmProviders enum")
        print(f"   Provider value: {LlmProviders.AGENTCORE.value}")
    else:
        print("❌ AGENTCORE not found in LlmProviders enum")
        return False

    # Check models_by_provider mapping
    if "agentcore" in litellm.models_by_provider:
        print("✅ agentcore found in models_by_provider")
        print(f"   Supported models: {litellm.models_by_provider['agentcore']}")
    else:
        print("❌ agentcore not found in models_by_provider")
        return False

    return True

def test_message_transformation():
    """Test message transformation to AgentCore format"""
    print("\n🔄 Testing Message Transformation")
    print("=" * 50)

    config = AgentCoreConfig()

    # Test simple message
    messages = [
        {"role": "user", "content": "Hello, world!"}
    ]

    try:
        agentcore_request = config._transform_messages_to_agentcore(messages)
        print("✅ Simple message transformation successful")
        print(f"   Request format: {json.dumps(agentcore_request, indent=2)}")

        # Validate required fields
        if "prompt" in agentcore_request and "runtimeSessionId" in agentcore_request:
            print("✅ Required fields present (prompt, runtimeSessionId)")

            # Check session ID length (should be >= 33 chars)
            session_id = agentcore_request["runtimeSessionId"]
            if len(session_id) >= 33:
                print(f"✅ Session ID length valid: {len(session_id)} chars")
            else:
                print(f"❌ Session ID too short: {len(session_id)} chars (need >= 33)")
                return False
        else:
            print("❌ Missing required fields")
            return False

    except Exception as e:
        print(f"❌ Message transformation failed: {e}")
        return False

    # Test conversation with history
    messages_with_history = [
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"}
    ]

    try:
        agentcore_request = config._transform_messages_to_agentcore(messages_with_history)
        print("✅ Conversation history transformation successful")

        if "context" in agentcore_request:
            print("✅ Context field present for conversation history")
            print(f"   Context: {agentcore_request['context']}")
        else:
            print("❌ Context field missing for conversation history")
            return False

    except Exception as e:
        print(f"❌ Conversation transformation failed: {e}")
        return False

    return True

def test_model_parsing():
    """Test model string parsing"""
    print("\n🏷️  Testing Model Parsing")
    print("=" * 50)

    config = AgentCoreConfig()

    test_cases = [
        ("simple_conversation_agent-py20Ve6ZUA/v1", True),
        ("agent-123/live", True),
        ("agent/alias/extra", False)  # Only this should fail (too many parts)
    ]

    for model_str, should_succeed in test_cases:
        try:
            result = config._parse_model(model_str)
            agent_id = result.get("agent_name") or result.get("arn")
            alias_id = result.get("qualifier")
            if should_succeed:
                print(f"✅ {model_str} -> agent_id: {agent_id}, alias_id: {alias_id}")
            else:
                print(f"❌ {model_str} should have failed but didn't")
                return False
        except ValueError as e:
            if not should_succeed:
                print(f"✅ {model_str} correctly failed: {e}")
            else:
                print(f"❌ {model_str} should have succeeded: {e}")
                return False

    return True

def test_arn_building():
    """Test agent ARN construction"""
    print("\n🏗️  Testing ARN Building")
    print("=" * 50)

    config = AgentCoreConfig()

    # Test ARN building
    agent_id = "simple_conversation_agent-py20Ve6ZUA"
    region = "eu-central-1"

    arn = config._build_agent_arn(agent_id, region)
    # ARN format: arn:aws:bedrock-agentcore:region:account:runtime/agent-name
    # Account ID will be dynamically fetched, just check structure
    if arn.startswith(f"arn:aws:bedrock-agentcore:{region}:") and arn.endswith(f":runtime/{agent_id}"):
        print(f"✅ ARN built correctly: {arn}")
    else:
        print(f"❌ ARN mismatch. Expected: {expected_arn}, Got: {arn}")
        return False

    return True

def test_response_transformation():
    """Test AgentCore response transformation to LiteLLM format"""
    print("\n📤 Testing Response Transformation")
    print("=" * 50)

    config = AgentCoreConfig()

    # Mock AgentCore response
    agentcore_response = {
        "response": "Hello! You said: Hello, world!. I'm a simple conversation agent running on AgentCore Runtime!",
        "metadata": {
            "prompt_tokens": 10,
            "completion_tokens": 25
        }
    }

    try:
        model_response = config._transform_agentcore_to_litellm(
            agentcore_response=agentcore_response,
            model="bedrock/agentcore/simple_conversation_agent-py20Ve6ZUA/v1",
            created_at=1234567890
        )

        print("✅ Response transformation successful")
        print(f"   Response ID: {model_response.id}")
        print(f"   Model: {model_response.model}")
        print(f"   Content: {model_response.choices[0].message.content}")
        print(f"   Usage: prompt={model_response.usage.prompt_tokens}, completion={model_response.usage.completion_tokens}")

        # Validate structure
        if (model_response.choices and
            len(model_response.choices) > 0 and
            model_response.choices[0].message and
            model_response.usage):
            print("✅ Response structure valid")
        else:
            print("❌ Response structure invalid")
            return False

    except Exception as e:
        print(f"❌ Response transformation failed: {e}")
        return False

    return True

def main():
    """Run all tests"""
    print("🧪 AgentCore Provider Validation Tests")
    print("=" * 60)

    tests = [
        ("Provider Registration", test_provider_registration),
        ("Message Transformation", test_message_transformation),
        ("Model Parsing", test_model_parsing),
        ("ARN Building", test_arn_building),
        ("Response Transformation", test_response_transformation)
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"\n❌ {test_name} FAILED")
        except Exception as e:
            print(f"\n💥 {test_name} CRASHED: {e}")

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed! AgentCore provider is ready.")
        return True
    else:
        print("⚠️  Some tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)