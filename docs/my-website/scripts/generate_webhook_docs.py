#!/usr/bin/env python3
# ruff: noqa: T201, PLR0915
"""
Auto-generate documentation for generic API compatible callbacks.

This script reads the generic_api_compatible_callbacks.json file and generates
individual documentation pages for each callback that has `docs.show_in_docs: true`.

Usage:
    python generate_webhook_docs.py

The generated docs will be placed in:
    docs/my-website/docs/observability/webhook_integrations/
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict


# Paths relative to this script
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent  # workspace root
JSON_PATH = PROJECT_ROOT / "litellm" / "integrations" / "generic_api" / "generic_api_compatible_callbacks.json"
OUTPUT_DIR = SCRIPT_DIR.parent / "docs" / "observability" / "webhook_integrations"


def load_callbacks() -> Dict[str, Any]:
    """Load the generic API compatible callbacks JSON file."""
    with open(JSON_PATH, "r") as f:
        return json.load(f)


def get_env_var_description(env_var: str, callback_name: str) -> str:
    """Generate a description for an environment variable based on naming conventions."""
    env_var_lower = env_var.lower()
    
    if "url" in env_var_lower or "endpoint" in env_var_lower:
        return "Webhook endpoint URL"
    elif "key" in env_var_lower or "token" in env_var_lower or "secret" in env_var_lower:
        return "API key or authentication token"
    else:
        return "Configuration value"


def generate_doc_page(callback_name: str, config: Dict[str, Any]) -> str:
    """Generate markdown documentation for a callback."""
    docs = config.get("docs", {})
    display_name = docs.get("display_name", callback_name.replace("_", " ").title())
    description = docs.get("description", f"Send LLM logs to {display_name}.")
    website = docs.get("website", "")
    additional_notes = docs.get("additional_notes", "")
    
    env_vars = config.get("environment_variables", [])
    event_types = config.get("event_types", ["llm_api_success", "llm_api_failure"])
    headers = config.get("headers", {})
    
    # Build the markdown content
    lines = []
    
    # Header
    lines.append(f"# {display_name}")
    lines.append("")
    
    # Description
    lines.append(description)
    if website:
        lines.append(f" [Learn more about {display_name}]({website}).")
    lines.append("")
    
    # Note about auto-generation
    lines.append(":::info Auto-Generated Documentation")
    lines.append("")
    lines.append("This documentation is auto-generated from [`generic_api_compatible_callbacks.json`](https://github.com/BerriAI/litellm/blob/main/litellm/integrations/generic_api/generic_api_compatible_callbacks.json).")
    lines.append("")
    lines.append(":::")
    lines.append("")
    
    # Event Types
    lines.append("## Supported Events")
    lines.append("")
    for event_type in event_types:
        emoji = "✅" if event_type == "llm_api_success" else "❌"
        event_display = "Success events" if event_type == "llm_api_success" else "Failure events"
        lines.append(f"- {emoji} **{event_display}** (`{event_type}`)")
    lines.append("")
    
    # Environment Variables
    lines.append("## Environment Variables")
    lines.append("")
    lines.append("| Variable | Description | Required |")
    lines.append("|----------|-------------|----------|")
    for env_var in env_vars:
        desc = get_env_var_description(env_var, callback_name)
        lines.append(f"| `{env_var}` | {desc} | Yes |")
    lines.append("")
    
    # Additional notes if provided
    if additional_notes:
        lines.append(":::tip Setup Note")
        lines.append("")
        lines.append(additional_notes)
        lines.append("")
        lines.append(":::")
        lines.append("")
    
    # Quick Start - Proxy
    lines.append("## Quick Start")
    lines.append("")
    lines.append("### LiteLLM Proxy (config.yaml)")
    lines.append("")
    lines.append("```yaml")
    lines.append("model_list:")
    lines.append("  - model_name: gpt-4")
    lines.append("    litellm_params:")
    lines.append("      model: openai/gpt-4")
    lines.append("      api_key: os.environ/OPENAI_API_KEY")
    lines.append("")
    lines.append("litellm_settings:")
    lines.append(f'  callbacks: ["{callback_name}"]')
    lines.append("")
    lines.append("environment_variables:")
    for env_var in env_vars:
        lines.append(f"  {env_var}: \"your-value-here\"")
    lines.append("```")
    lines.append("")
    
    # Start proxy command
    lines.append("Start the proxy:")
    lines.append("")
    lines.append("```bash")
    lines.append("litellm --config config.yaml")
    lines.append("```")
    lines.append("")
    
    # Test request
    lines.append("Test with a request:")
    lines.append("")
    lines.append("```bash")
    lines.append("curl -X POST http://localhost:4000/chat/completions \\")
    lines.append("  -H \"Content-Type: application/json\" \\")
    lines.append("  -H \"Authorization: Bearer sk-1234\" \\")
    lines.append("  -d '{")
    lines.append('    "model": "gpt-4",')
    lines.append('    "messages": [{"role": "user", "content": "Hello!"}]')
    lines.append("  }'")
    lines.append("```")
    lines.append("")
    
    # Quick Start - SDK
    lines.append("### LiteLLM Python SDK")
    lines.append("")
    lines.append("```python")
    lines.append("import os")
    lines.append("import litellm")
    lines.append("from litellm import completion")
    lines.append("")
    lines.append("# Set environment variables")
    for env_var in env_vars:
        lines.append(f'os.environ["{env_var}"] = "your-value-here"')
    lines.append('os.environ["OPENAI_API_KEY"] = "your-openai-key"')
    lines.append("")
    lines.append("# Enable the callback")
    lines.append(f'litellm.success_callback = ["{callback_name}"]')
    if "llm_api_failure" in event_types:
        lines.append(f'litellm.failure_callback = ["{callback_name}"]')
    lines.append("")
    lines.append("# Make a request - logs will be sent automatically")
    lines.append("response = completion(")
    lines.append('    model="gpt-4",')
    lines.append('    messages=[{"role": "user", "content": "Hello!"}]')
    lines.append(")")
    lines.append("print(response)")
    lines.append("```")
    lines.append("")
    
    # Payload format reference
    lines.append("## Logged Payload")
    lines.append("")
    lines.append("The [LiteLLM Standard Logging Payload](https://docs.litellm.ai/docs/proxy/logging_spec) is sent to your endpoint. This includes:")
    lines.append("")
    lines.append("- Request ID and timestamps")
    lines.append("- Model and provider information")
    lines.append("- Token usage and cost")
    lines.append("- Request/response content (unless redacted)")
    lines.append("- Metadata and user information")
    lines.append("")
    
    # Headers sent
    if headers:
        lines.append("## Request Headers")
        lines.append("")
        lines.append("The following headers are sent with each request:")
        lines.append("")
        lines.append("| Header | Value |")
        lines.append("|--------|-------|")
        for header_key, header_value in headers.items():
            # Mask sensitive values in display
            display_value = header_value
            if "{{environment_variables." in header_value:
                display_value = header_value.replace("{{environment_variables.", "`$").replace("}}", "`")
            lines.append(f"| `{header_key}` | {display_value} |")
        lines.append("")
    
    # Contributing note
    lines.append("## See Also")
    lines.append("")
    lines.append("- [Generic API Callback](../generic_api.md) - Custom webhook configuration")
    lines.append("- [Logging Spec](../../proxy/logging_spec.md) - Payload format details")
    lines.append("- [Contribute Custom Webhook API](../../contribute_integration/custom_webhook_api.md) - Add your own integration")
    lines.append("")
    
    return "\n".join(lines)


def generate_index_page(callbacks: Dict[str, Dict[str, Any]]) -> str:
    """Generate the index page listing all webhook integrations."""
    lines = []
    
    lines.append("---")
    lines.append("sidebar_label: Overview")
    lines.append("---")
    lines.append("")
    lines.append("# Webhook Integrations")
    lines.append("")
    lines.append("LiteLLM supports sending logs to various webhook-based services. These integrations use the Generic API Callback system and require minimal configuration.")
    lines.append("")
    lines.append(":::info Auto-Generated")
    lines.append("")
    lines.append("This page is auto-generated from [`generic_api_compatible_callbacks.json`](https://github.com/BerriAI/litellm/blob/main/litellm/integrations/generic_api/generic_api_compatible_callbacks.json).")
    lines.append("")
    lines.append(":::")
    lines.append("")
    lines.append("## Available Integrations")
    lines.append("")
    lines.append("| Integration | Description | Events |")
    lines.append("|-------------|-------------|--------|")
    
    for callback_name, config in sorted(callbacks.items()):
        docs = config.get("docs", {})
        if not docs.get("show_in_docs", False):
            continue
            
        display_name = docs.get("display_name", callback_name.replace("_", " ").title())
        description = docs.get("description", f"Send logs to {display_name}")
        # Truncate description for table
        if len(description) > 80:
            description = description[:77] + "..."
        
        event_types = config.get("event_types", [])
        events_str = ", ".join(event_types)
        
        lines.append(f"| [{display_name}](./{callback_name}.md) | {description} | `{events_str}` |")
    
    lines.append("")
    lines.append("## Quick Setup")
    lines.append("")
    lines.append("All webhook integrations follow the same pattern:")
    lines.append("")
    lines.append("1. Set the required environment variables")
    lines.append("2. Add the callback name to your config")
    lines.append("3. Start making requests")
    lines.append("")
    lines.append("```yaml")
    lines.append("litellm_settings:")
    lines.append('  callbacks: ["callback_name"]')
    lines.append("")
    lines.append("environment_variables:")
    lines.append("  CALLBACK_WEBHOOK_URL: \"https://...\"")
    lines.append("  CALLBACK_API_KEY: \"sk-...\"  # if required")
    lines.append("```")
    lines.append("")
    lines.append("## Adding New Integrations")
    lines.append("")
    lines.append("Want to add support for a new webhook service? See the [Contributing Guide](../../contribute_integration/custom_webhook_api.md).")
    lines.append("")
    lines.append("Just add your service to `generic_api_compatible_callbacks.json` and the documentation will be auto-generated!")
    lines.append("")
    
    return "\n".join(lines)


def main():
    """Main entry point."""
    print(f"Loading callbacks from: {JSON_PATH}")
    
    if not JSON_PATH.exists():
        print(f"ERROR: JSON file not found: {JSON_PATH}")
        sys.exit(1)
    
    callbacks = load_callbacks()
    print(f"Found {len(callbacks)} callbacks")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Track which files we generate
    generated_files = []
    
    # Generate individual pages
    for callback_name, config in callbacks.items():
        docs = config.get("docs", {})
        if not docs.get("show_in_docs", False):
            print(f"  Skipping {callback_name} (show_in_docs=false)")
            continue
        
        display_name = docs.get("display_name", callback_name)
        output_path = OUTPUT_DIR / f"{callback_name}.md"
        
        content = generate_doc_page(callback_name, config)
        with open(output_path, "w") as f:
            f.write(content)
        
        print(f"  Generated: {output_path.name} ({display_name})")
        generated_files.append(callback_name)
    
    # Generate index page
    index_path = OUTPUT_DIR / "index.md"
    index_content = generate_index_page(callbacks)
    with open(index_path, "w") as f:
        f.write(index_content)
    print("  Generated: index.md")
    
    # Generate sidebar items file (can be imported in sidebars.js)
    sidebar_items = [f"observability/webhook_integrations/{name}" for name in sorted(generated_files)]
    sidebar_path = OUTPUT_DIR / "_sidebar_items.json"
    with open(sidebar_path, "w") as f:
        json.dump(sidebar_items, f, indent=2)
    print("  Generated: _sidebar_items.json")
    
    print(f"\nDone! Generated {len(generated_files)} integration docs.")
    print("\nTo include in sidebar, update sidebars.js to include:")
    print('  {')
    print('    type: "category",')
    print('    label: "Webhook Integrations",')
    print('    items: [')
    print('      "observability/webhook_integrations/index",')
    for name in sorted(generated_files):
        print(f'      "observability/webhook_integrations/{name}",')
    print('    ]')
    print('  }')


if __name__ == "__main__":
    main()
