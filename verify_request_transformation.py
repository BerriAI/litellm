"""
Verify that the request transformation includes output_format and the correct beta header.
This doesn't make actual API calls - it just validates the transformation logic.
"""
import json
from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    AnthropicMessagesConfig,
)
from litellm.types.llms.anthropic import ANTHROPIC_BETA_HEADER_VALUES


def test_request_transformation():
    """Verify request transformation includes output_format."""
    config = AnthropicMessagesConfig()

    print("=" * 80)
    print("Verifying Request Transformation Logic")
    print("=" * 80)

    # Define output format
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

    messages = [
        {
            "role": "user",
            "content": "Extract info: John Smith (john@example.com) wants Enterprise plan."
        }
    ]

    # Test 1: Check supported parameters
    print("\n1️⃣  Checking supported parameters...")
    print("-" * 80)
    supported_params = config.get_supported_anthropic_messages_params(
        model="claude-sonnet-4-5-20250929"
    )
    print(f"Supported parameters: {supported_params}")

    if "output_format" in supported_params:
        print("✅ output_format is in supported parameters list")
    else:
        print("❌ output_format is NOT in supported parameters list - FIX FAILED!")
        return False

    # Test 2: Transform request with output_format
    print("\n2️⃣  Testing request transformation with output_format...")
    print("-" * 80)

    optional_params = {
        "max_tokens": 1024,
        "temperature": 0.7,
        "output_format": output_format
    }

    transformed_request = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-5-20250929",
        messages=messages,
        anthropic_messages_optional_request_params=optional_params.copy(),
        litellm_params={},
        headers={}
    )

    print(f"Transformed request keys: {list(transformed_request.keys())}")

    if "output_format" in transformed_request:
        print("✅ output_format is in transformed request")
        print(f"\noutput_format content:")
        print(json.dumps(transformed_request["output_format"], indent=2))

        # Verify the structure
        of = transformed_request["output_format"]
        if of.get("type") == "json_schema" and "schema" in of:
            print("✅ output_format has correct structure (type: json_schema, schema: {...})")
        else:
            print("❌ output_format structure is incorrect")
            return False
    else:
        print("❌ output_format is NOT in transformed request - FIX FAILED!")
        return False

    # Test 3: Check beta header injection
    print("\n3️⃣  Testing beta header injection...")
    print("-" * 80)

    headers = {}
    optional_params_with_output = {
        "output_format": output_format
    }

    updated_headers = config._update_headers_with_anthropic_beta(
        headers=headers,
        optional_params=optional_params_with_output
    )

    print(f"Headers after injection: {updated_headers}")

    if "anthropic-beta" in updated_headers:
        beta_value = updated_headers["anthropic-beta"]
        print(f"✅ anthropic-beta header present: {beta_value}")

        expected_beta = ANTHROPIC_BETA_HEADER_VALUES.STRUCTURED_OUTPUT_2025_09_25.value
        if expected_beta in beta_value:
            print(f"✅ Correct beta header value '{expected_beta}' found")
        else:
            print(f"❌ Expected beta value '{expected_beta}' NOT found in: {beta_value}")
            return False
    else:
        print("❌ anthropic-beta header NOT added - FIX FAILED!")
        return False

    # Test 4: Check beta header merging with existing headers
    print("\n4️⃣  Testing beta header merging with existing headers...")
    print("-" * 80)

    headers_with_existing = {
        "anthropic-beta": "custom-beta-feature"
    }

    optional_params_multi = {
        "output_format": output_format,
        "context_management": {"type": "ephemeral"}
    }

    merged_headers = config._update_headers_with_anthropic_beta(
        headers=headers_with_existing.copy(),
        optional_params=optional_params_multi
    )

    beta_value = merged_headers.get("anthropic-beta", "")
    print(f"Merged beta header: {beta_value}")

    # Check all expected values are present
    expected_values = [
        "custom-beta-feature",
        "structured-outputs-2025-11-13",
        "context-management-2025-06-27"
    ]

    all_present = all(val in beta_value for val in expected_values)
    if all_present:
        print(f"✅ All expected beta values present: {expected_values}")
    else:
        print(f"❌ Not all expected values present")
        for val in expected_values:
            if val in beta_value:
                print(f"  ✅ {val}")
            else:
                print(f"  ❌ {val} - MISSING!")
        return False

    # Test 5: Full request simulation
    print("\n5️⃣  Full request simulation...")
    print("-" * 80)

    full_optional_params = {
        "max_tokens": 1024,
        "output_format": output_format,
        "temperature": 0.7
    }

    headers_for_request = {}
    headers_for_request = config._update_headers_with_anthropic_beta(
        headers=headers_for_request,
        optional_params=full_optional_params
    )

    request_body = config.transform_anthropic_messages_request(
        model="claude-sonnet-4-5-20250929",
        messages=messages,
        anthropic_messages_optional_request_params=full_optional_params.copy(),
        litellm_params={},
        headers=headers_for_request
    )

    print("\nSimulated HTTP Request:")
    print("-" * 80)
    print("Headers:")
    for key, value in headers_for_request.items():
        print(f"  {key}: {value}")

    print("\nRequest Body (JSON):")
    print(json.dumps(request_body, indent=2))

    # Verify the complete request
    if "output_format" in request_body and "anthropic-beta" in headers_for_request:
        if "structured-outputs-2025-11-13" in headers_for_request["anthropic-beta"]:
            print("\n✅ Complete request looks correct!")
            print("   - output_format is in request body")
            print("   - structured-outputs beta header is set")
            return True

    print("\n❌ Complete request is missing required elements")
    return False


if __name__ == "__main__":
    print("🔍 Verifying Structured Outputs Fix\n")

    success = test_request_transformation()

    print("\n" + "=" * 80)
    if success:
        print("✅ ALL VERIFICATIONS PASSED - Fix is working correctly!")
        print("=" * 80)
        print("\nThe fix ensures:")
        print("  1. output_format is accepted as a valid parameter")
        print("  2. output_format is preserved in the request body")
        print("  3. structured-outputs-2025-11-13 beta header is auto-injected")
        print("  4. Beta headers merge correctly with existing headers")
        print("\nReady to test against real Anthropic API!")
    else:
        print("❌ VERIFICATION FAILED - Fix may not be working correctly")
        print("=" * 80)

    exit(0 if success else 1)
