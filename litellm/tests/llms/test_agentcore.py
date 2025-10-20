#!/usr/bin/env python3
"""
Test script to validate the AgentCore provider implementation
without requiring a deployed agent.
"""

import sys
import os
import json
import logging

# Add the parent directory to sys.path to import our AgentCore provider
sys.path.insert(0, os.path.dirname(__file__))

import litellm
from litellm.llms.bedrock.agentcore import AgentCoreConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def test_provider_registration():
    """Test that AgentCore provider is properly registered with LiteLLM"""
    logger.info("🔍 Testing AgentCore Provider Registration")
    logger.info("=" * 50)

    # Check if agentcore is in the supported providers
    from litellm.types.utils import LlmProviders

    if hasattr(LlmProviders, "AGENTCORE"):
        logger.info("✅ AGENTCORE found in LlmProviders enum")
        logger.info(f"   Provider value: {LlmProviders.AGENTCORE.value}")
    else:
        logger.error("❌ AGENTCORE not found in LlmProviders enum")
        return False

    # Check models_by_provider mapping
    if "agentcore" in litellm.models_by_provider:
        logger.info("✅ agentcore found in models_by_provider")
        logger.info(f"   Supported models: {litellm.models_by_provider['agentcore']}")
    else:
        logger.error("❌ agentcore not found in models_by_provider")
        return False

    return True


def test_message_transformation():
    """Test message transformation to AgentCore format"""
    logger.info("\n🔄 Testing Message Transformation")
    logger.info("=" * 50)

    config = AgentCoreConfig()

    # Test simple message
    messages = [{"role": "user", "content": "Hello, world!"}]

    try:
        agentcore_request = config._transform_messages_to_agentcore(messages)
        logger.info("✅ Simple message transformation successful")
        logger.info(f"   Request format: {json.dumps(agentcore_request, indent=2)}")

        # Validate required fields
        if "prompt" in agentcore_request and "runtimeSessionId" in agentcore_request:
            logger.info("✅ Required fields present (prompt, runtimeSessionId)")

            # Check session ID length (should be >= 33 chars)
            session_id = agentcore_request["runtimeSessionId"]
            if len(session_id) >= 33:
                logger.info(f"✅ Session ID length valid: {len(session_id)} chars")
            else:
                logger.error(
                    f"❌ Session ID too short: {len(session_id)} chars (need >= 33)"
                )
                return False
        else:
            logger.error("❌ Missing required fields")
            return False

    except Exception as e:
        logger.error(f"❌ Message transformation failed: {e}")
        return False

    # Test conversation with history
    messages_with_history = [
        {"role": "user", "content": "What's 2+2?"},
        {"role": "assistant", "content": "2+2 equals 4."},
        {"role": "user", "content": "What about 3+3?"},
    ]

    try:
        agentcore_request = config._transform_messages_to_agentcore(
            messages_with_history
        )
        logger.info("✅ Conversation history transformation successful")

        if "context" in agentcore_request:
            logger.info("✅ Context field present for conversation history")
            logger.info(f"   Context: {agentcore_request['context']}")
        else:
            logger.error("❌ Context field missing for conversation history")
            return False

    except Exception as e:
        logger.error(f"❌ Conversation transformation failed: {e}")
        return False

    return True


def test_model_parsing():
    """Test model string parsing"""
    logger.info("\n🏷️  Testing Model Parsing")
    logger.info("=" * 50)

    config = AgentCoreConfig()

    test_cases = [
        ("simple_conversation_agent-py20Ve6ZUA/v1", True),
        ("agent-123/live", True),
        ("agent/alias/extra", False),  # Only this should fail (too many parts)
    ]

    for model_str, should_succeed in test_cases:
        try:
            result = config._parse_model(model_str)
            agent_id = result.get("agent_name") or result.get("arn")
            alias_id = result.get("qualifier")
            if should_succeed:
                logger.info(
                    f"✅ {model_str} -> agent_id: {agent_id}, alias_id: {alias_id}"
                )
            else:
                logger.error(f"❌ {model_str} should have failed but didn't")
                return False
        except ValueError as e:
            if not should_succeed:
                logger.info(f"✅ {model_str} correctly failed: {e}")
            else:
                logger.error(f"❌ {model_str} should have succeeded: {e}")
                return False

    return True


def test_arn_building():
    """Test agent ARN construction"""
    logger.info("\n🏗️  Testing ARN Building")
    logger.info("=" * 50)

    config = AgentCoreConfig()

    # Test ARN building
    agent_id = "simple_conversation_agent-py20Ve6ZUA"
    region = "eu-central-1"

    arn = config._build_agent_arn(agent_id, region)
    # ARN format: arn:aws:bedrock-agentcore:region:account:runtime/agent-name
    # Account ID will be dynamically fetched, just check structure
    if arn.startswith(f"arn:aws:bedrock-agentcore:{region}:") and arn.endswith(
        f":runtime/{agent_id}"
    ):
        logger.info(f"✅ ARN built correctly: {arn}")
    else:
        logger.error(f"❌ ARN mismatch. Got: {arn}")
        return False

    return True


def test_response_transformation():
    """Test AgentCore response transformation to LiteLLM format"""
    logger.info("\n📤 Testing Response Transformation")
    logger.info("=" * 50)

    config = AgentCoreConfig()

    # Mock AgentCore response
    agentcore_response = {
        "response": "Hello! You said: Hello, world!. I'm a simple conversation agent running on AgentCore Runtime!",
        "metadata": {"prompt_tokens": 10, "completion_tokens": 25},
    }

    try:
        model_response = config._transform_agentcore_to_litellm(
            agentcore_response=agentcore_response,
            model="bedrock/agentcore/simple_conversation_agent-py20Ve6ZUA/v1",
            created_at=1234567890,
        )

        logger.info("✅ Response transformation successful")
        logger.info(f"   Response ID: {model_response.id}")
        logger.info(f"   Model: {model_response.model}")
        logger.info(f"   Content: {model_response.choices[0].message.content}")
        logger.info(
            f"   Usage: prompt={model_response.usage.prompt_tokens}, completion={model_response.usage.completion_tokens}"
        )

        # Validate structure
        if (
            model_response.choices
            and len(model_response.choices) > 0
            and model_response.choices[0].message
            and model_response.usage
        ):
            logger.info("✅ Response structure valid")
        else:
            logger.error("❌ Response structure invalid")
            return False

    except Exception as e:
        logger.error(f"❌ Response transformation failed: {e}")
        return False

    return True


def main():
    """Run all tests"""
    logger.info("🧪 AgentCore Provider Validation Tests")
    logger.info("=" * 60)

    tests = [
        ("Provider Registration", test_provider_registration),
        ("Message Transformation", test_message_transformation),
        ("Model Parsing", test_model_parsing),
        ("ARN Building", test_arn_building),
        ("Response Transformation", test_response_transformation),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                logger.error(f"\n❌ {test_name} FAILED")
        except Exception as e:
            logger.error(f"\n💥 {test_name} CRASHED: {e}")

    logger.info(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        logger.info("🎉 All tests passed! AgentCore provider is ready.")
        return True
    else:
        logger.warning("⚠️  Some tests failed. Check implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
