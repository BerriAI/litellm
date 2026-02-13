"""
Code coverage test to ensure all provider folders are documented.

This script validates that:
1. Every provider folder in litellm/llms/ has a corresponding entry in provider_endpoints_support.json
2. Every provider in litellm/llms/openai_like/providers.json is documented in provider_endpoints_support.json
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


class UndocumentedProviderError(Exception):
    """Raised when providers are found without documentation."""

    pass


# Special folders that should be excluded from validation
EXCLUDED_FOLDERS = {
    "__pycache__",
    "base_llm",
    "deprecated_providers",
    "custom_httpx",
    "pass_through",
    "openai_like",  # This is a generic handler, not a specific provider
    "aiohttp_openai",  # Internal implementation detail for async HTTP
}


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


def get_llm_provider_folders() -> Set[str]:
    """Get all provider folder names from litellm/llms directory."""
    repo_root = get_repo_root()
    llms_dir = repo_root / "litellm" / "llms"

    if not llms_dir.exists():
        print(f"❌ ERROR: Could not find llms directory at {llms_dir}")
        sys.exit(1)

    folders = set()
    for item in llms_dir.iterdir():
        if item.is_dir() and item.name not in EXCLUDED_FOLDERS:
            folders.add(item.name)

    return folders


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


def get_openai_like_providers() -> Set[str]:
    """Get all provider names from litellm/llms/openai_like/providers.json."""
    repo_root = get_repo_root()
    providers_file = repo_root / "litellm" / "llms" / "openai_like" / "providers.json"

    if not providers_file.exists():
        print(
            f"⚠️  WARNING: Could not find openai_like/providers.json at {providers_file}"
        )
        return set()

    with open(providers_file, "r") as f:
        data = json.load(f)

    # Return all provider keys from the JSON
    return set(data.keys())


def get_documented_providers(data: Dict) -> Set[str]:
    """Get all provider slugs documented in provider_endpoints_support.json."""
    providers = data.get("providers", {})

    # Get all provider keys, including those with slashes
    documented = set()
    for provider_key in providers.keys():
        # For providers like "azure_ai/doc-intelligence", extract base name
        base_name = provider_key.split("/")[0]
        documented.add(base_name)
        # Also add the full key in case folder name matches exactly
        documented.add(provider_key)

    return documented


def normalize_provider_name(folder_name: str) -> Set[str]:
    """
    Generate possible provider names that might match a folder.

    Some folders might have variations in the JSON:
    - github_copilot folder -> github_copilot provider
    - azure folder -> azure, azure_text, azure_ai providers
    """
    variations = {folder_name}

    # Add common variations
    if "_" in folder_name:
        # Try without underscores (though less common)
        variations.add(folder_name.replace("_", ""))

    return variations


def main():
    """Main function to validate provider documentation."""
    print("🔍 Checking that all providers are documented...")

    has_errors = False

    # Check 1: Provider folders in litellm/llms
    print("\n📁 Checking provider folders in litellm/llms/...")
    provider_folders = get_llm_provider_folders()
    print(f"✓ Found {len(provider_folders)} provider folders")

    # Check 2: OpenAI-like providers
    print("\n📋 Checking openai_like providers...")
    openai_like_providers = get_openai_like_providers()
    print(f"✓ Found {len(openai_like_providers)} openai_like providers")

    # Load the JSON file
    data = load_provider_endpoints_file()
    documented_providers = get_documented_providers(data)
    print(
        f"\n✓ Found {len(data.get('providers', {}))} provider entries in provider_endpoints_support.json"
    )

    # Check for undocumented folders
    undocumented_folders = []
    for folder in sorted(provider_folders):
        # Check if folder name or any variation is documented
        variations = normalize_provider_name(folder)
        if not any(var in documented_providers for var in variations):
            undocumented_folders.append(folder)

    # Check for undocumented openai_like providers
    undocumented_openai_like = []
    for provider in sorted(openai_like_providers):
        # Generate multiple possible variations of the provider name
        variations = {
            provider,  # Original name (e.g., "nano-gpt")
            provider.replace(
                "-", "_"
            ),  # Replace hyphens with underscores (e.g., "nano_gpt")
            provider.replace("-", ""),  # Remove hyphens (e.g., "nanogpt")
            provider.replace("_", ""),  # Remove underscores
        }

        # Special case mappings for known variations
        special_mappings = {
            "veniceai": "venice",
            "nano-gpt": "nanogpt",
        }
        if provider in special_mappings:
            variations.add(special_mappings[provider])

        # Check if any variation is documented
        if not any(var in documented_providers for var in variations):
            undocumented_openai_like.append(provider)

    # Collect all error messages
    error_messages: List[str] = []

    # Report errors for undocumented folders
    if undocumented_folders:
        has_errors = True
        error_msg = "\n❌ ERROR: The following provider folders are not documented:\n"
        error_msg += "=" * 70 + "\n"
        for folder in undocumented_folders:
            error_msg += f"  - litellm/llms/{folder}/\n"

        error_msg += "\n" + "=" * 70 + "\n"
        error_msg += f"\n💡 To fix: Add entries for these {len(undocumented_folders)} provider(s)\n"
        error_msg += (
            "   in the 'providers' section of provider_endpoints_support.json\n"
        )
        error_msg += "\nExample format:\n"
        error_msg += '  "providers": {\n'
        for folder in undocumented_folders[:3]:
            error_msg += f'    "{folder}": {{\n'
            error_msg += f'      "display_name": "{folder.replace("_", " ").title()} (`{folder}`)",\n'
            error_msg += (
                f'      "url": "https://docs.litellm.ai/docs/providers/{folder}",\n'
            )
            error_msg += '      "endpoints": {\n'
            error_msg += '        "chat_completions": true,\n'
            error_msg += '        "messages": true,\n'
            error_msg += '        "responses": true,\n'
            error_msg += '        "embeddings": false,\n'
            error_msg += "        ...\n"
            error_msg += "      }\n"
            error_msg += "    },\n"
        if len(undocumented_folders) > 3:
            error_msg += "    ...\n"
        error_msg += "  }\n"

        print(error_msg)
        error_messages.append(
            f"Found {len(undocumented_folders)} undocumented provider folders: {', '.join(undocumented_folders)}"
        )

    # Report errors for undocumented openai_like providers
    if undocumented_openai_like:
        has_errors = True
        error_msg = (
            "\n❌ ERROR: The following openai_like providers are not documented:\n"
        )
        error_msg += "=" * 70 + "\n"
        for provider in undocumented_openai_like:
            error_msg += f"  - {provider}\n"

        error_msg += "\n" + "=" * 70 + "\n"
        error_msg += f"\n💡 To fix: Add entries for these {len(undocumented_openai_like)} provider(s)\n"
        error_msg += (
            "   in the 'providers' section of provider_endpoints_support.json\n"
        )
        error_msg += "\nExample format:\n"
        error_msg += '  "providers": {\n'
        for provider in undocumented_openai_like[:3]:
            normalized = provider.replace("-", "_")
            error_msg += f'    "{normalized}": {{\n'
            error_msg += f'      "display_name": "{provider.replace("-", " ").replace("_", " ").title()} (`{normalized}`)",\n'
            error_msg += (
                f'      "url": "https://docs.litellm.ai/docs/providers/{normalized}",\n'
            )
            error_msg += '      "endpoints": {\n'
            error_msg += '        "chat_completions": true,\n'
            error_msg += '        "messages": true,\n'
            error_msg += '        "responses": true,\n'
            error_msg += '        "embeddings": false,\n'
            error_msg += "        ...\n"
            error_msg += "      }\n"
            error_msg += "    },\n"
        if len(undocumented_openai_like) > 3:
            error_msg += "    ...\n"
        error_msg += "  }\n"

        print(error_msg)
        error_messages.append(
            f"Found {len(undocumented_openai_like)} undocumented openai_like providers: {', '.join(undocumented_openai_like)}"
        )

    # Raise exception if there are any errors
    if has_errors:
        error_summary = " AND ".join(error_messages)
        raise UndocumentedProviderError(
            f"Provider documentation validation failed: {error_summary}"
        )

    print(f"\n✅ All {len(provider_folders)} provider folders are documented!")
    print(f"✅ All {len(openai_like_providers)} openai_like providers are documented!")
    print("\n🎉 All provider documentation checks passed!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except UndocumentedProviderError as e:
        print(f"\n🚨 CRITICAL ERROR: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n🚨 UNEXPECTED ERROR: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)
