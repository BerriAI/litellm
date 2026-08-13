"""
Code coverage test to ensure all endpoints documented in sidebars.js are defined in provider_endpoints_support.json.

This script:
1. Extracts all endpoint entries from the "Supported Endpoints" section of sidebars.js
2. Validates that each endpoint has a corresponding entry in the "endpoints" object of provider_endpoints_support.json
3. Checks that the "docs_label" field is present in each endpoint definition
"""

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class MissingEndpointDefinitionError(Exception):
    """Raised when endpoints are documented in sidebars.js but missing from provider_endpoints_support.json."""

    pass


def get_repo_root() -> Path:
    """Get the repository root directory."""
    # Check if litellm directory exists in current working directory
    cwd = Path.cwd()
    if (cwd / "litellm").exists() and (cwd / "litellm").is_dir():
        # We're already at the repo root
        return cwd

    # Otherwise, navigate up from script location
    current = Path(__file__).resolve()
    # Navigate up from tests/code_coverage_tests/
    return current.parent.parent.parent


def extract_endpoints_from_sidebars() -> Dict[str, str]:
    """
    Extract endpoint entries from sidebars.js.

    Returns a dict mapping endpoint_key -> label
    Only extracts top-level endpoint entries from the "Supported Endpoints" section.
    """
    repo_root = get_repo_root()
    sidebars_path = repo_root / "docs" / "my-website" / "sidebars.js"

    if not sidebars_path.exists():
        print(f"❌ ERROR: Could not find sidebars.js at {sidebars_path}")
        sys.exit(1)

    with open(sidebars_path, "r") as f:
        content = f.read()

    # Find the Supported Endpoints section
    supported_start = content.find('label: "Supported Endpoints"')
    if supported_start == -1:
        print("⚠️  WARNING: Could not find 'Supported Endpoints' section")
        return {}

    # Find the items array within this section
    items_start = content.find("items: [", supported_start)
    if items_start == -1:
        print("⚠️  WARNING: Could not find items array in Supported Endpoints")
        return {}

    # Find the end of this items array
    # Look for the closing ], at the same indentation level
    items_end = content.find("\n      ],\n    },\n    {", items_start)
    if items_end == -1:
        items_end = content.find("\n      ],\n    }", items_start)

    section = content[items_start:items_end]

    endpoints = {}

    # Pattern 1: Categories with labels at the top level (8 spaces indent)
    # Example:  "        {type: "category", label: "/a2a - A2A Agent Gateway""
    category_pattern = (
        r'^\s{8}\{\s*\n\s{10}type:\s*"category",\s*\n\s{10}label:\s*"([^"]+)"'
    )
    for match in re.finditer(category_pattern, section, re.MULTILINE):
        label = match.group(1)
        # Skip utility categories
        if "Pass-through" in label or label == "Vertex AI":
            continue
        endpoint_key = label.split(" - ")[0].strip("/").replace("/", "_")
        endpoints[endpoint_key] = label

    # Pattern 2: Standalone doc strings at top level (8 spaces indent)
    # Example: "        "assistants","
    standalone_pattern = r'^\s{8}"([a-zA-Z_][a-zA-Z0-9_]*)",?\s*$'
    for match in re.finditer(standalone_pattern, section, re.MULTILINE):
        doc_id = match.group(1)
        endpoints[doc_id] = doc_id

    return endpoints


def load_provider_endpoints_file() -> Dict:
    """Load the provider_endpoints_support.json file."""
    repo_root = get_repo_root()
    file_path = repo_root / "provider_endpoints_support.json"

    if not file_path.exists():
        print(
            f"❌ ERROR: Could not find provider_endpoints_support.json at {file_path}"
        )
        sys.exit(1)

    with open(file_path, "r") as f:
        return json.load(f)


def get_defined_endpoints(data: Dict) -> Dict[str, Dict]:
    """Get all endpoint definitions from provider_endpoints_support.json."""
    return data.get("endpoints", {})


def normalize_endpoint_key(key: str) -> Set[str]:
    """
    Generate variations of an endpoint key for matching.

    Examples:
    - "a2a" -> {"a2a"}
    - "chat_completions" -> {"chat_completions", "chatcompletions"}
    - "vector_stores" -> {"vector_stores", "vectorstores"}
    """
    variations = {key, key.replace("_", "")}
    return variations


def check_provider_endpoint_keys(data: Dict) -> List[str]:
    """
    Check that all endpoint keys used in providers are defined in the root endpoints section.

    Returns a list of missing endpoint keys.
    """
    # Collect all unique endpoint keys used across all providers
    provider_endpoint_keys = set()
    providers = data.get("providers", {})

    for provider_name, provider_data in providers.items():
        if "endpoints" in provider_data and isinstance(
            provider_data["endpoints"], dict
        ):
            provider_endpoint_keys.update(provider_data["endpoints"].keys())

    # Get all endpoint definitions
    defined_endpoints = data.get("endpoints", {})

    # Collect all provider_json_field values from endpoint definitions
    provider_json_fields = set()
    for endpoint_key, endpoint_data in defined_endpoints.items():
        if isinstance(endpoint_data, dict) and "provider_json_field" in endpoint_data:
            provider_json_fields.add(endpoint_data["provider_json_field"])

    # Find missing endpoint keys
    missing_keys = []
    for key in sorted(provider_endpoint_keys):
        if key not in provider_json_fields:
            missing_keys.append(key)

    return missing_keys


def check_unused_endpoints(data: Dict) -> List[Tuple[str, str]]:
    """
    Check that all defined endpoints are used by at least one provider.

    Returns a list of tuples (endpoint_key, provider_json_field) for unused endpoints.
    """
    # Special endpoints that don't need to be used by specific providers
    # These are utility/framework endpoints available across the platform
    SPECIAL_ENDPOINTS = {
        "apply_guardrail",  # Guardrail application - works across providers
        "mcp",  # Model Context Protocol - works across providers
    }

    # Get all endpoint definitions
    defined_endpoints = data.get("endpoints", {})
    providers = data.get("providers", {})

    # Collect all endpoint keys used by providers
    used_keys = set()
    for provider_data in providers.values():
        if "endpoints" in provider_data and isinstance(
            provider_data["endpoints"], dict
        ):
            used_keys.update(provider_data["endpoints"].keys())

    # Find unused endpoints (excluding special ones)
    unused = []
    for endpoint_key, endpoint_data in defined_endpoints.items():
        # Skip special endpoints
        if endpoint_key in SPECIAL_ENDPOINTS:
            continue

        if isinstance(endpoint_data, dict) and "provider_json_field" in endpoint_data:
            provider_json_field = endpoint_data["provider_json_field"]
            # Check if this provider_json_field is used by any provider
            if provider_json_field not in used_keys:
                unused.append((endpoint_key, provider_json_field))

    return sorted(unused)


def main():
    """Main function to validate endpoint coverage."""
    print(
        "🔍 Checking endpoint coverage between sidebars.js and provider_endpoints_support.json..."
    )

    has_errors = False

    # Load provider_endpoints_support.json
    data = load_provider_endpoints_file()
    defined_endpoints = get_defined_endpoints(data)

    # Test 1: Check that endpoints from sidebars.js have docs_label entries
    print("\n📖 Test 1: Checking endpoints from sidebars.js...")
    sidebar_endpoints = extract_endpoints_from_sidebars()
    print(f"✓ Found {len(sidebar_endpoints)} endpoints in sidebars.js")
    print(
        f"✓ Found {len(defined_endpoints)} endpoint definitions in provider_endpoints_support.json"
    )

    # Check for missing endpoints
    missing_endpoints = []

    # Collect all docs_label values from defined endpoints
    defined_docs_labels = set()
    for endpoint_data in defined_endpoints.values():
        if isinstance(endpoint_data, dict) and "docs_label" in endpoint_data:
            defined_docs_labels.add(endpoint_data["docs_label"])

    for sidebar_key, sidebar_label in sorted(sidebar_endpoints.items()):
        # Generate variations for matching against docs_label
        variations = normalize_endpoint_key(sidebar_key)

        # Check if any variation exists in docs_label values
        if not any(var in defined_docs_labels for var in variations):
            missing_endpoints.append((sidebar_key, sidebar_label))

    # Report missing endpoints from sidebars
    if missing_endpoints:
        has_errors = True
        error_msg = "\n❌ ERROR: The following endpoints are in sidebars.js but missing from provider_endpoints_support.json:\n"
        error_msg += "=" * 70 + "\n"

        for key, label in missing_endpoints:
            error_msg += f"  - {key}\n"
            error_msg += f'    Label in sidebars.js: "{label}"\n'

        error_msg += "\n" + "=" * 70 + "\n"
        error_msg += f"\n💡 To fix: Add these {len(missing_endpoints)} endpoint(s) to the 'endpoints' object\n"
        error_msg += "   in provider_endpoints_support.json\n"
        error_msg += "\nExample format:\n"
        error_msg += '  "endpoints": {\n'

        for key, label in missing_endpoints[:5]:
            error_msg += f'    "{key}": {{\n'
            error_msg += f'      "docs_label": "{label}",\n'
            error_msg += f'      "provider_json_field": "{key}",\n'
            error_msg += f'      "description": "Description of the {label} endpoint"\n'
            error_msg += "    },\n"

        if len(missing_endpoints) > 5:
            error_msg += "    ...\n"

        error_msg += "  }\n"

        print(error_msg)
    else:
        print(
            f"✅ All {len(sidebar_endpoints)} endpoints from sidebars.js are defined!"
        )

    # Test 2: Check that all provider endpoint keys have provider_json_field entries
    print("\n📋 Test 2: Checking provider endpoint keys...")
    missing_provider_keys = check_provider_endpoint_keys(data)

    if missing_provider_keys:
        has_errors = True
        error_msg = "\n❌ ERROR: The following endpoint keys are used in providers but missing provider_json_field definitions:\n"
        error_msg += "=" * 70 + "\n"

        for key in missing_provider_keys:
            # Find which providers use this key
            using_providers = []
            for provider_name, provider_data in data.get("providers", {}).items():
                if key in provider_data.get("endpoints", {}):
                    using_providers.append(provider_name)

            error_msg += f"  - {key}\n"
            error_msg += f"    Used by {len(using_providers)} provider(s): {', '.join(using_providers[:3])}"
            if len(using_providers) > 3:
                error_msg += f" and {len(using_providers) - 3} more"
            error_msg += "\n"

        error_msg += "\n" + "=" * 70 + "\n"
        error_msg += f"\n💡 To fix: Add these {len(missing_provider_keys)} endpoint(s) to the 'endpoints' object\n"
        error_msg += "   in provider_endpoints_support.json with 'provider_json_field' matching the key\n"
        error_msg += "\nExample format:\n"
        error_msg += '  "endpoints": {\n'

        for key in missing_provider_keys[:3]:
            error_msg += f'    "{key}": {{\n'
            error_msg += f'      "docs_label": "{key}",\n'
            error_msg += f'      "provider_json_field": "{key}",\n'
            error_msg += f'      "description": "Description of the {key} endpoint"\n'
            error_msg += "    },\n"

        if len(missing_provider_keys) > 3:
            error_msg += "    ...\n"

        error_msg += "  }\n"

        print(error_msg)
    else:
        print("✅ All provider endpoint keys have provider_json_field definitions!")

    # Test 3: Check that all defined endpoints are used by at least one provider
    print("\n🔍 Test 3: Checking for unused endpoint definitions...")
    unused_endpoints = check_unused_endpoints(data)

    if unused_endpoints:
        has_errors = True
        error_msg = "\n⚠️  WARNING: The following endpoint definitions are not used by any provider:\n"
        error_msg += "=" * 70 + "\n"

        for endpoint_key, provider_json_field in unused_endpoints:
            endpoint_data = defined_endpoints.get(endpoint_key, {})
            docs_label = endpoint_data.get("docs_label", "N/A")
            error_msg += f"  - {endpoint_key}\n"
            error_msg += f"    provider_json_field: '{provider_json_field}'\n"
            error_msg += f"    docs_label: '{docs_label}'\n"

        error_msg += "\n" + "=" * 70 + "\n"
        error_msg += f"\n💡 These {len(unused_endpoints)} endpoint(s) are defined but not used by any provider.\n"
        error_msg += "   Either:\n"
        error_msg += (
            "   1. Add the endpoint to relevant providers' 'endpoints' objects, OR\n"
        )
        error_msg += "   2. Remove the endpoint definition if it's no longer needed\n"

        print(error_msg)
    else:
        print("✅ All endpoint definitions are used by at least one provider!")

    # Raise error if any tests failed
    if has_errors:
        error_summary = []
        if missing_endpoints:
            error_summary.append(f"{len(missing_endpoints)} endpoints from sidebars.js")
        if missing_provider_keys:
            error_summary.append(f"{len(missing_provider_keys)} provider endpoint keys")
        if unused_endpoints:
            error_summary.append(f"{len(unused_endpoints)} unused endpoint definitions")

        raise MissingEndpointDefinitionError(
            f"Endpoint validation failed: Missing definitions for {' and '.join(error_summary)}"
        )

    print("\n🎉 All endpoint coverage validations passed!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MissingEndpointDefinitionError as e:
        print(f"\n🚨 CRITICAL ERROR: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n🚨 UNEXPECTED ERROR: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
