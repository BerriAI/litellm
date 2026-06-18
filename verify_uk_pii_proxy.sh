#!/bin/bash
# Manual verification script for UK PII entity types in litellm proxy
# Tests that UK entities are properly recognized and masked by Presidio

set -e

echo "=== UK PII Entity Types Verification Script ==="
echo ""
echo "This script verifies that UK_PASSPORT, UK_POSTCODE, and UK_VEHICLE_REGISTRATION"
echo "entity types are properly recognized by the Presidio guardrail."
echo ""

CONFIG_FILE="/tmp/uk_pii_test_config.yaml"

cat > "$CONFIG_FILE" <<'EOF'
model_list:
  - model_name: gpt-3.5-turbo
    litellm_params:
      model: gpt-3.5-turbo
      api_key: os.environ/OPENAI_API_KEY

guardrails:
  - guardrail_name: "uk-pii-test"
    litellm_params:
      guardrail: presidio
      mode: pre_call
      default_on: true
      pii_entities:
        UK_NHS: MASK
        UK_NINO: MASK
        UK_PASSPORT: MASK
        UK_POSTCODE: MASK
        UK_VEHICLE_REGISTRATION: MASK
EOF

echo "Created test config at: $CONFIG_FILE"
echo ""
echo "Starting litellm proxy on port 4000 with UK PII guardrail enabled..."
echo ""
echo "Run the following commands in separate terminals to test:"
echo ""
echo "# Terminal 1: Start proxy"
echo "python litellm/proxy/proxy_cli.py --config $CONFIG_FILE --detailed_debug"
echo ""
echo "# Terminal 2: Test UK_PASSPORT"
echo "curl -X POST http://localhost:4000/chat/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\": \"gpt-3.5-turbo\", \"messages\": [{\"role\": \"user\", \"content\": \"My passport is 012345678\"}]}' | jq"
echo ""
echo "# Terminal 2: Test UK_POSTCODE"
echo "curl -X POST http://localhost:4000/chat/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\": \"gpt-3.5-turbo\", \"messages\": [{\"role\": \"user\", \"content\": \"I live at SW1A 1AA\"}]}' | jq"
echo ""
echo "# Terminal 2: Test UK_VEHICLE_REGISTRATION"
echo "curl -X POST http://localhost:4000/chat/completions \\"
echo "  -H 'Content-Type: application/json' \\"
echo "  -d '{\"model\": \"gpt-3.5-turbo\", \"messages\": [{\"role\": \"user\", \"content\": \"My car registration is AB12 CDE\"}]}' | jq"
echo ""
echo "Expected: The sensitive UK data should be masked (replaced with <UK_PASSPORT>, <UK_POSTCODE>, <UK_VEHICLE_REGISTRATION>)"
echo "          in the request before being sent to OpenAI."
