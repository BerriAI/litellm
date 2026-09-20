# Prompt caching request table proof

Source: `93eb7c7264e8b17e366c717b068e0d0caaceb96f`; base: `f49fd22875b10d968fdefda1a5c3d6e0b7717f2c`

Screenshots show the live dashboard before and after the feature. The payloads are the exact synthetic requests used against a real Anthropic provider through the gateway account

Configure the local `cache-proof` model alias for `anthropic/claude-sonnet-5`, enable `enable_anthropic_prompt_caching`, and connect a local Postgres database. The recorded run used a local forwarding relay to the authenticated gateway Messages endpoint. Supply your own authorized provider credentials

Run the three payloads in order with the curl loop in the PR. The first writes the stable prefix, the second reuses it, and the third supplies its own cache controls. `requests.json` records the resulting request-table response at the source commit above

The zero-cost rows below these three requests in the screenshot are earlier failed provider attempts, which still recorded LiteLLM injection
