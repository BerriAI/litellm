"""Test the Chat Completions Bridge tool search example from docs (line 856-887)"""
import os
from dotenv import load_dotenv
load_dotenv()

import litellm

try:
    response = litellm.completion(
        model="openai/responses/gpt-5.4",
        messages=[{"role": "user", "content": "Look up invoice INV-2024-001"}],
        tools=[
            {"type": "tool_search"},
            {
                "type": "namespace",
                "name": "billing",
                "description": "Billing and invoicing tools",
                "tools": [
                    {
                        "type": "function",
                        "name": "get_invoice",
                        "description": "Get an invoice by ID",
                        "parameters": {
                            "type": "object",
                            "properties": {"invoice_id": {"type": "string"}},
                            "required": ["invoice_id"],
                        },
                        "defer_loading": True,
                    },
                ],
            },
        ],
    )

    print("=== Raw response ===")
    print(f"tool_calls value: {response.choices[0].message.tool_calls}")
    print(f"tool_calls is None? {response.choices[0].message.tool_calls is None}")
    print()

    # Test the docs code exactly as written
    print("=== Testing docs code (no None guard) ===")
    try:
        for tool_call in response.choices[0].message.tool_calls:
            print(f"Called: {tool_call.function.name}({tool_call.function.arguments})")
    except TypeError as e:
        print(f"  !!! TypeError: {e}")
        print(f"  Greptile was RIGHT - need 'or []' guard")

    # Test with the fix
    print()
    print("=== Testing with fix (or [] guard) ===")
    for tool_call in (response.choices[0].message.tool_calls or []):
        print(f"Called: {tool_call.function.name}({tool_call.function.arguments})")
    print("  OK - no crash")

except Exception as e:
    print(f"API Error: {type(e).__name__}: {e}")
