#!/usr/bin/env python3
"""
Script to update the README.md providers table from provider_endpoints_support.json
"""

import json
import re
from pathlib import Path

# Define paths
REPO_ROOT = Path(__file__).parent.parent
JSON_PATH = REPO_ROOT / "provider_endpoints_support.json"
README_PATH = REPO_ROOT / "README.md"

# Endpoint column headers
ENDPOINT_COLUMNS = [
    ("/chat/completions", "chat_completions"),
    ("/messages", "messages"),
    ("/responses", "responses"),
    ("/embeddings", "embeddings"),
    ("/image/generations", "image_generations"),
    ("/audio/transcriptions", "audio_transcriptions"),
    ("/audio/speech", "audio_speech"),
    ("/moderations", "moderations"),
    ("/batches", "batches"),
    ("/rerank", "rerank"),
]


def load_providers_data():
    """Load provider data from JSON file"""
    with open(JSON_PATH, 'r') as f:
        data = json.load(f)
    
    # Handle both old and new format
    if "providers" in data:
        return data["providers"]
    return data


def generate_markdown_table(providers_data):
    """Generate markdown table from providers data"""
    
    # Sort providers alphabetically by display name
    sorted_providers = sorted(
        providers_data.items(),
        key=lambda x: x[1]['display_name'].lower()
    )
    
    # Generate header
    header_cols = ["Provider"] + [col[0] for col in ENDPOINT_COLUMNS]
    header = "| " + " | ".join(header_cols) + " |"
    separator = "|" + "|".join(["-" * (len(col) + 2) for col in header_cols]) + "|"
    
    # Generate rows
    rows = []
    for slug, data in sorted_providers:
        display_name = data['display_name']
        url = data['url']
        
        # Build row
        row_parts = [f"[{display_name}]({url})"]
        
        for _, endpoint_key in ENDPOINT_COLUMNS:
            supported = data['endpoints'].get(endpoint_key, False)
            row_parts.append("✅" if supported else "")
        
        row = "| " + " | ".join(row_parts) + " |"
        rows.append(row)
    
    # Combine all parts
    table_lines = [
        "<!-- AUTO-GENERATED TABLE - DO NOT EDIT MANUALLY -->",
        "<!-- Edit provider_endpoints_support.json and run scripts/update_readme_providers_table.py -->",
        "",
        header,
        separator
    ] + rows + [
        "<!-- END AUTO-GENERATED TABLE -->"
    ]
    
    return "\n".join(table_lines)


def update_readme(table_markdown):
    """Update README.md with new table"""
    with open(README_PATH, 'r') as f:
        content = f.read()
    
    print(f"  Original README length: {len(content)} bytes")
    
    # Find the table section
    # Look for the AUTO-GENERATED comment or the header, and replace until Read the Docs
    pattern = r"(## Supported Providers.*?\n\n)(?:<!-- AUTO-GENERATED TABLE.*?<!-- END AUTO-GENERATED TABLE -->|.*?)(\n\n\[\*\*Read the Docs\*\*\])"
    
    # Test if pattern matches
    match = re.search(pattern, content, flags=re.DOTALL)
    if not match:
        print("❌ Pattern did not match in README.md")
        return False
    
    print(f"  Pattern matched, replacing table...")
    
    def replacer(match):
        return match.group(1) + table_markdown + match.group(2)
    
    new_content = re.sub(pattern, replacer, content, flags=re.DOTALL)
    
    print(f"  New README length: {len(new_content)} bytes")
    
    if new_content == content:
        print("  ℹ️  Table is already up-to-date, no changes needed")
        return True  # Not an error - table is already correct
    
    with open(README_PATH, 'w') as f:
        f.write(new_content)
    
    print("  ✓ README.md has been updated")
    return True


def main():
    """Main function"""
    print("Loading provider data from provider_endpoints_support.json...")
    providers_data = load_providers_data()
    print(f"✓ Loaded {len(providers_data)} providers")
    
    print("\nGenerating markdown table...")
    table_markdown = generate_markdown_table(providers_data)
    print(f"✓ Generated table with {len(providers_data)} rows")
    
    print("\nUpdating README.md...")
    if update_readme(table_markdown):
        print("✓ Successfully updated README.md")
        print("\n📝 Please review the changes and commit both files:")
        print("   - provider_endpoints_support.json")
        print("   - README.md")
    else:
        print("❌ Failed to update README.md")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

