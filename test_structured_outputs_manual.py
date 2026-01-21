"""
Manual test script to validate structured outputs fix against real Anthropic API.

Usage:
    ANTHROPIC_API_KEY=your-key-here poetry run python test_structured_outputs_manual.py
"""
import os
import sys
import json
from litellm import anthropic_messages
import asyncio


async def test_structured_outputs():
    """Test structured outputs with the /v1/messages endpoint."""

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("❌ ANTHROPIC_API_KEY not set. Please set it to run this test.")
        sys.exit(1)

    print("=" * 80)
    print("Testing Structured Outputs with /v1/messages endpoint")
    print("=" * 80)

    # Test message
    messages = [
        {
            "role": "user",
            "content": "Extract the key information from this email: John Smith (john@example.com) is interested in our Enterprise plan and wants to schedule a demo for next Tuesday at 2pm."
        }
    ]

    # Define the output schema
    output_format = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "plan_interest": {"type": "string"},
                "demo_requested": {"type": "boolean"}
            },
            "required": ["name", "email", "plan_interest", "demo_requested"],
            "additionalProperties": False
        }
    }

    print("\n1️⃣  Testing WITH output_format (should return JSON)...")
    print("-" * 80)

    try:
        response_with_format = await anthropic_messages.acreate(
            model="anthropic/claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=messages,
            output_format=output_format,
            api_key=api_key,
        )

        print(f"✅ Response received!")
        print(f"Response type: {response_with_format.get('type')}")
        print(f"Model: {response_with_format.get('model')}")
        print(f"Stop reason: {response_with_format.get('stop_reason')}")

        # Extract content
        content = response_with_format.get('content', [])
        if content:
            first_content = content[0]
            content_type = first_content.get('type')
            text = first_content.get('text', '')

            print(f"\nContent type: {content_type}")
            print(f"Content text:\n{text}")

            # Try to parse as JSON to verify it's actually JSON
            try:
                parsed = json.loads(text)
                print(f"\n✅ Successfully parsed as JSON!")
                print(f"Parsed data: {json.dumps(parsed, indent=2)}")

                # Verify it has the expected structure
                expected_keys = {"name", "email", "plan_interest", "demo_requested"}
                actual_keys = set(parsed.keys())
                if actual_keys == expected_keys:
                    print(f"✅ Response has correct schema!")
                else:
                    print(f"⚠️  Schema mismatch. Expected: {expected_keys}, Got: {actual_keys}")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse as JSON: {e}")
                print("This indicates the fix may not be working correctly.")

        print(f"\nUsage: {response_with_format.get('usage')}")

    except Exception as e:
        print(f"❌ Error with output_format: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("\n2️⃣  Testing WITHOUT output_format (baseline - will return markdown)...")
    print("-" * 80)

    try:
        response_without_format = await anthropic_messages.acreate(
            model="anthropic/claude-sonnet-4-5-20250929",
            max_tokens=1024,
            messages=messages,
            api_key=api_key,
        )

        print(f"✅ Response received!")

        # Extract content
        content = response_without_format.get('content', [])
        if content:
            first_content = content[0]
            text = first_content.get('text', '')

            print(f"Content text:\n{text}")

            # This should NOT be JSON, it should be markdown
            try:
                json.loads(text)
                print(f"⚠️  Unexpectedly got JSON (should be markdown)")
            except json.JSONDecodeError:
                print(f"\n✅ Correctly returned non-JSON text (markdown format)")

    except Exception as e:
        print(f"❌ Error without output_format: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)


async def test_bedrock_structured_outputs():
    """Test structured outputs with Bedrock provider."""

    print("\n\n")
    print("=" * 80)
    print("Testing Structured Outputs with BEDROCK via /v1/messages endpoint")
    print("=" * 80)

    # Check for AWS credentials
    aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")
    aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    aws_region = os.getenv("AWS_REGION_NAME", "us-east-1")

    if not aws_access_key or not aws_secret_key:
        print("⚠️  AWS credentials not set. Skipping Bedrock test.")
        print("   Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY to test Bedrock.")
        return

    messages = [
        {
            "role": "user",
            "content": "Extract the key information from this email: John Smith (john@example.com) is interested in our Enterprise plan."
        }
    ]

    output_format = {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "plan_interest": {"type": "string"}
            },
            "required": ["name", "email", "plan_interest"],
            "additionalProperties": False
        }
    }

    print("\nTesting Bedrock with output_format...")
    print("-" * 80)

    try:
        response = await anthropic_messages.acreate(
            model="bedrock/anthropic.claude-sonnet-4-5-v2:0",
            max_tokens=1024,
            messages=messages,
            output_format=output_format,
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            aws_region_name=aws_region,
        )

        print(f"✅ Response received!")

        content = response.get('content', [])
        if content:
            text = content[0].get('text', '')
            print(f"Content:\n{text}")

            try:
                parsed = json.loads(text)
                print(f"\n✅ Successfully parsed as JSON: {json.dumps(parsed, indent=2)}")
            except json.JSONDecodeError as e:
                print(f"❌ Failed to parse as JSON: {e}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🧪 Manual Validation Test for Structured Outputs Fix")
    print()

    # Run the tests
    asyncio.run(test_structured_outputs())
    asyncio.run(test_bedrock_structured_outputs())
