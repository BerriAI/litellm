# Bedrock Messages Interview Task

Complete the AWS Bedrock implementation behind the Anthropic-compatible
`POST /v1/messages` route. The gateway plumbing, provider registration, bearer
authentication setup, SigV4 support scaffolding, streaming response plumbing,
test harness, and live harness are already present. The remaining work is to
complete the shared Messages abstractions, Bedrock request and response
transforms, and Bedrock EventStream decoding.

## Given

The gateway is configured to use:

- `AWS_BEARER_TOKEN_BEDROCK` for the live bearer credential
- Region `us-west-2`
- Model `us.anthropic.claude-sonnet-4-5-20250929-v1:0`

Claude 3.5 and Claude 3.7 Bedrock model identifiers are end of life and should
not be used.

Bedrock streaming responses use
`application/vnd.amazon.eventstream`, not ordinary SSE. The EventStream
payload contains an object shaped like
`{"bytes": "<base64-encoded Anthropic event JSON>"}`. The
`aws-smithy-eventstream::MessageFrameDecoder` type is available for parsing
the binary framing.

## Running the gateway and live harness

From `litellm-rust/`, start the gateway with:

```bash
cargo run -p litellm-ai-gateway --features server
```

In another shell, with `AWS_BEARER_TOKEN_BEDROCK` available in the
environment, run:

```bash
./bedrock_messages_harness.sh
```

The harness exercises simple non-streaming output, streaming output, tool use,
and an invalid-model response.

## Tests and checks

Format and compile the workspace with:

```bash
cargo fmt --check
cargo build
```

Run the existing test suite with:

```bash
cargo test --workspace
```

The Bedrock-specific tests in the core transformation and messages route
modules describe the expected behavior. They are intentionally failing until
the implementation is completed.

## Required scenarios

Make all of these scenarios pass:

1. Non-streaming Bedrock InvokeModel requests use the correct encoded model
   path, request body shape, authentication headers, and response mapping
2. Streaming Bedrock EventStream frames, including frames split across
   transport chunks, become normalized Anthropic SSE
3. Tool-use requests and responses round trip correctly
4. Bedrock HTTP errors and EventStream exception frames map to the gateway's
   expected error shapes
