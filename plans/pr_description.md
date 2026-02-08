# Fix Gemini function_response role for tool results

## Summary
- Add the missing `role="user"` on tool/function response content parts sent to Gemini.
- Prevents `INVALID_ARGUMENT` errors from the Vertex AI / Gemini REST API during multi-turn tool calling.

## Problem
Gemini rejects requests where `function_response` parts are missing a `role` in the `contents` array. The current transformation logic appends tool response parts without `role="user"`, causing 400 `INVALID_ARGUMENT` failures when a tool response is followed by a subsequent message.

## Changes
- Set `role="user"` when appending tool call response content in the Gemini transformation path.

## Evidence / Reproduction
- Error log: `error-v1beta-models-gemini-3-pro-preview-streamGenerateContent-2026-02-08T114448-d2e51a48.log`
- Sample request: `request.json`

### Repro Steps
1. Send a chat with Gemini that triggers a tool/function call.
2. Provide the tool result in the next turn.
3. Continue the conversation with another message.
4. Observe 400 `INVALID_ARGUMENT` when the tool response content lacks `role`.

## Testing
- Not run (poetry is not available in the environment).
