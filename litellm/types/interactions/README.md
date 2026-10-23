# Interactions API Types

This directory contains type definitions for the Google Interactions API.

## Generated Types

The `generated.py` file is auto-generated from the official OpenAPI spec:
https://ai.google.dev/static/api/interactions.openapi.json

### How to Regenerate

The compliance tests in `tests/unit/interactions/test_openapi_compliance.py` read the snapshot at
`tests/unit/interactions/interactions.openapi.json`, so a PR never depends on Google's live spec. The scheduled
`interactions-spec-canary` workflow runs those tests against the live spec once a day (`--interactions-spec-url`),
files the issue "Interactions spec canary is red" on a red run and closes it once a run is green again. A red run
means the spec moved in a way these types no longer cover

When the red run names one new status, the small fix is a hand edit adding that member to the enum the failing
schema's model carries, the way `queued` was added in #34135. When the spec moved further than that, refresh the
snapshot and regenerate the types from it:

```bash
curl -fsSL https://ai.google.dev/static/api/interactions.openapi.json \
    -o tests/unit/interactions/interactions.openapi.json
uv tool run --from datamodel-code-generator==0.66.0 datamodel-codegen \
    --input tests/unit/interactions/interactions.openapi.json \
    --input-file-type openapi \
    --output litellm/types/interactions/generated.py \
    --output-model-type pydantic_v2.BaseModel \
    --target-python-version 3.10 \
    --base-class litellm.types.llms.base.LiteLLMBaseModel
```

The pin is the same `datamodel-code-generator` version `scripts/generate_trace_types.py` uses. The generator names
inline enums by order of appearance (`Status`, `Status1`, `Status2`, ...), so a regeneration can rename them. The
compliance tests resolve each status enum through its model's `status` field, but
`tests/unit/interactions/test_background_cost_polling.py` and
`tests/unit/proxy/hooks/test_proxy_track_cost_callback.py` import `Status1` by name and need the new name, and
`litellm/types/interactions/__init__.py` exports the interaction status enum as `InteractionStatus`, which has to
point at the enum the regenerated `Interaction.status` carries (a compliance test checks their members match). The
spec has grown a lot since the 2025-12-16 generation (agents, triggers, credentials, environments), so a
regeneration today is a review-sized change rather than a refresh

Then add the LiteLLM-specific types at the bottom of the generated file:
- `InteractionsAPIResponse`
- `InteractionsAPIStreamingResponse`
- `DeleteInteractionResult`
- `CancelInteractionResult`

### Key Types

**Request Types:**
- `CreateModelInteractionParams` - For model interactions
- `CreateAgentInteractionParams` - For agent interactions

**Content Types:**
- `Content` - Union of all content types (text, image, audio, etc.)
- `TextContent` - Text content with `type: "text"`
- `Turn` - A turn in multi-turn conversation with `role` and `content`

**Tool Types:**
- `Tool` - Union of all tool types
- `Function` - Function tool declaration

**Response Types:**
- `InteractionsAPIResponse` - LiteLLM response wrapper
- `InteractionsAPIStreamingResponse` - Streaming response chunk
