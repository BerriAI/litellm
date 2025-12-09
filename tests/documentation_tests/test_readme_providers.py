"""
Test to ensure all providers in LlmProviders enum are documented in README.md
"""

import os
import re
from litellm.types.utils import LlmProviders

# Define paths
readme_path = "./README.md"

# Providers that shouldn't be required in README
# (specialized tools, observability, database providers that aren't LLM providers)
EXCLUDED_PROVIDERS = {
    "aiohttp_openai",  # internal http variant
    "langfuse",  # observability, not LLM provider
    "humanloop",  # observability, not LLM provider
    "pg_vector",  # database, not LLM provider
    "dotprompt",  # prompt management, not provider
    "vertex_ai_beta",  # beta variant, not needed in main table
}

def get_enum_providers():
    """
    Get all provider values from LlmProviders enum, excluding internal variants.
    """
    providers = set()
    for provider in LlmProviders:
        provider_value = provider.value
        if provider_value not in EXCLUDED_PROVIDERS:
            providers.add(provider_value)
    return providers


def get_readme_providers():
    """
    Extract provider slugs from README.md provider table.
    
    Looks for provider slugs in backticks within parentheses, e.g.:
    [OpenAI (`openai`)](url) -> extracts "openai"
    """
    providers = set()
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find the supported providers table
        # Look for the section starting with "## Supported Providers"
        providers_section = re.search(
            r"## Supported Providers.*?\n\n(.*?)(?=\n##|\Z)",
            content,
            re.DOTALL | re.MULTILINE,
        )

        if providers_section:
            table_content = providers_section.group(1)
            # Extract provider slugs from backticks in parentheses: (`slug`)
            provider_slug_pattern = re.compile(r"\(`([^`]+)`\)")
            matches = provider_slug_pattern.findall(table_content)
            providers.update(matches)
        else:
            raise Exception("Could not find 'Supported Providers' section in README.md")

    except FileNotFoundError:
        raise Exception(f"README.md not found at {readme_path}")
    except Exception as e:
        raise Exception(f"Error reading README.md: {e}")

    return providers


def get_readme_provider_names():
    """
    Extract provider display names from README.md provider table in order.
    
    Returns a list of provider names as they appear in the table.
    """
    provider_names = []
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find the supported providers table
        providers_section = re.search(
            r"## Supported Providers.*?\n\n(.*?)(?=\n\[|\n##|\Z)",
            content,
            re.DOTALL | re.MULTILINE,
        )

        if providers_section:
            table_content = providers_section.group(1)
            # Extract provider names from table rows that start with |
            # Split by lines and process each line
            for line in table_content.split('\n'):
                # Only process lines that are table rows (start with |)
                if line.strip().startswith('|') and '[' in line:
                    # Extract provider name from: | [Provider Name (...)](...) |
                    match = re.search(r'\|\s*\[([^\]]+)\]\(', line)
                    if match:
                        provider_name = match.group(1)
                        # Skip header row and separator row
                        if provider_name != "Provider" and not provider_name.startswith('-'):
                            provider_names.append(provider_name)
        else:
            raise Exception("Could not find 'Supported Providers' section in README.md")

    except FileNotFoundError:
        raise Exception(f"README.md not found at {readme_path}")
    except Exception as e:
        raise Exception(f"Error reading README.md: {e}")

    return provider_names


def test_all_providers_documented():
    """
    Test that all providers in LlmProviders enum are documented in README.md.
    
    Verifies that provider slugs in the enum match the slugs shown in backticks
    in the README provider table.
    """
    enum_providers = get_enum_providers()
    readme_providers = get_readme_providers()

    print(f"\nProvider slugs in LlmProviders enum (filtered): {sorted(enum_providers)}")
    print(f"\nProvider slugs in README.md: {sorted(readme_providers)}")

    # Find undocumented providers
    undocumented = enum_providers - readme_providers

    if undocumented:
        raise AssertionError(
            f"\nProvider slugs not documented in README.md: {sorted(undocumented)}\n"
            f"Please add these providers to the README.md provider table with their slugs in backticks.\n"
            f"Example: [Provider Name (`slug`)](url)"
        )
    else:
        print(f"\n✓ All {len(enum_providers)} provider slugs are documented in README.md")


def test_providers_alphabetically_ordered():
    """
    Test that providers in README.md are listed in alphabetical order.
    """
    provider_names = get_readme_provider_names()
    
    if not provider_names:
        raise AssertionError("No provider names found in README.md")
    
    # Create a sorted version for comparison
    sorted_names = sorted(provider_names, key=str.lower)
    
    print(f"\nFound {len(provider_names)} providers in README.md")
    
    # Check if the list is alphabetically ordered
    out_of_order = []
    for i, (actual, expected) in enumerate(zip(provider_names, sorted_names)):
        if actual != expected:
            out_of_order.append({
                "position": i + 1,
                "actual": actual,
                "expected": expected
            })
    
    if out_of_order:
        error_msg = "\nProviders are not in alphabetical order:\n"
        for item in out_of_order[:10]:  # Show first 10 issues
            error_msg += f"  Position {item['position']}: Found '{item['actual']}', expected '{item['expected']}'\n"
        if len(out_of_order) > 10:
            error_msg += f"  ... and {len(out_of_order) - 10} more issues\n"
        raise AssertionError(error_msg)
    else:
        print(f"✓ All {len(provider_names)} providers are in alphabetical order")


if __name__ == "__main__":
    test_all_providers_documented()
    test_providers_alphabetically_ordered()

