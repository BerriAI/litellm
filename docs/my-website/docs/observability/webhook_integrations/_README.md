# Webhook Integrations (Auto-Generated)

**⚠️ DO NOT EDIT FILES IN THIS DIRECTORY MANUALLY!**

These documentation files are auto-generated from:
`litellm/integrations/generic_api/generic_api_compatible_callbacks.json`

## How to Update

1. Edit the JSON file at `litellm/integrations/generic_api/generic_api_compatible_callbacks.json`
2. Run the generator script:
   ```bash
   cd docs/my-website
   python3 scripts/generate_webhook_docs.py
   ```
3. Commit both the JSON changes and the regenerated docs

## Adding a New Integration

Add your service to `generic_api_compatible_callbacks.json` with the `docs` field:

```json
{
    "your_service": {
        "event_types": ["llm_api_success", "llm_api_failure"],
        "endpoint": "{{environment_variables.YOUR_SERVICE_URL}}",
        "headers": {
            "Content-Type": "application/json"
        },
        "environment_variables": ["YOUR_SERVICE_URL"],
        "docs": {
            "show_in_docs": true,
            "display_name": "Your Service Name",
            "description": "What your service does.",
            "website": "https://your-service.com/",
            "additional_notes": "Optional setup notes."
        }
    }
}
```

Then run the generator script.

## Build Process

The `npm run build` and `npm start` commands automatically run the generator via npm `prebuild` and `prestart` hooks.
