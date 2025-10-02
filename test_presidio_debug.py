"""
Debug script to test if Presidio guardrail is being triggered
"""

# Test to verify call_type matching
call_type = "acompletion"

# Simulate what Presidio checks
from litellm.types.utils import CallTypes as LitellmCallTypes

print(f"Testing call_type: {call_type}")
print(f"LitellmCallTypes.completion.value: {LitellmCallTypes.completion.value}")
print(f"LitellmCallTypes.acompletion.value: {LitellmCallTypes.acompletion.value}")

# This is the actual check from Presidio (line 405-411)
if (
    call_type
    in [
        LitellmCallTypes.completion.value,
        LitellmCallTypes.acompletion.value,
    ]
    or call_type == "mcp_call"
):
    print("✓ Presidio WOULD process this call_type")
else:
    print("✗ Presidio WOULD NOT process this call_type")

# Test with other call_types
for test_call_type in ["completion", "text_completion", "mcp_call", "embeddings"]:
    if (
        test_call_type
        in [
            LitellmCallTypes.completion.value,
            LitellmCallTypes.acompletion.value,
        ]
        or test_call_type == "mcp_call"
    ):
        print(f"✓ {test_call_type} WOULD be processed")
    else:
        print(f"✗ {test_call_type} WOULD NOT be processed")
