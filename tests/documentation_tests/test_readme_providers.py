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


if __name__ == "__main__":
    test_all_providers_documented()

