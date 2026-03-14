"""Test the Responses API tool search example from docs (line 705-783)"""
import os
from dotenv import load_dotenv
load_dotenv()

import litellm
import json

# Define namespaces with deferred tools
tools = [
    {"type": "tool_search"},  # Enable tool search
    {
        "type": "namespace",
        "name": "crm",
        "description": "CRM tools for customer management",
        "tools": [
            {
                "type": "function",
                "name": "get_customer",
                "description": "Get customer details by ID",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "customer_id": {"type": "string"}
                    },
                    "required": ["customer_id"],
                },
                "defer_loading": True,
            },
            {
                "type": "function",
                "name": "list_customers",
                "description": "List customers with optional filters",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["active", "inactive"]},
                    },
                },
                "defer_loading": True,
            },
        ],
    },
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
                    "properties": {
                        "invoice_id": {"type": "string"}
                    },
                    "required": ["invoice_id"],
                },
                "defer_loading": True,
            },
        ],
    },
]

try:
    response = litellm.responses(
        model="openai/gpt-5.4",
        input="Look up invoice INV-2024-001 from the billing system",
        tools=tools,
    )

    print("=== Raw response.output ===")
    print(response.output)
    print()

    # Test the parsing code from the docs
    print("=== Parsing output items ===")
    for item in response.output:
        print(f"  item type: {type(item)}")
        if isinstance(item, dict):
            print(f"    dict keys: {item.keys()}")
            if item["type"] == "tool_search_call":
                print(f"Searched namespaces: {item['arguments']['paths']}")
            elif item["type"] == "tool_search_output":
                print(f"Loaded {len(item['tools'])} tool(s)")
            elif item["type"] == "function_call":
                print(f"Called: {item.get('namespace', '')}.{item['name']}({item['arguments']})")
        else:
            print(f"    object attrs: {dir(item)}")
            if item.type == "function_call":
                # Greptile says this will fail if namespace is missing
                print(f"  Has 'namespace' attr? {hasattr(item, 'namespace')}")
                try:
                    print(f"Called: {item.namespace}.{item.name}({item.arguments})")
                except AttributeError as e:
                    print(f"  !!! AttributeError: {e}")
                    print(f"  Greptile was RIGHT - need getattr fallback")

except Exception as e:
    print(f"API Error: {type(e).__name__}: {e}")
