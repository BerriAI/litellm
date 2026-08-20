# Task: serve AWS Bedrock on `POST /v1/messages`

The gateway in this workspace already serves the Anthropic Messages API for
Anthropic and Azure. Your job is to make it serve AWS Bedrock too, well enough
that the Claude Code CLI can point `ANTHROPIC_BASE_URL` at this gateway and
complete a turn against a Bedrock-hosted Claude

Claude Code always sends `stream: true`, so streaming is part of the job, not a
stretch goal

## What you are given

Credentials and target, already in your environment:

- `AWS_BEARER_TOKEN_BEDROCK`, a Bedrock API key. Bedrock accepts it as a plain
  `Authorization: Bearer <token>` header, so you do not need to implement SigV4
- Region `us-west-2`, model `us.anthropic.claude-sonnet-4-5-20250929-v1:0`.
  The Claude 3.5 and 3.7 Bedrock ids are end of life and return 404

Already wired up, so you should not need to touch it:

- provider registration, the `BEDROCK_MODEL` startup config, the bearer and
  SigV4 auth paths in `crates/ai-gateway/src/messages/{prepare,handler}.rs`,
  and the streaming plumbing in `crates/ai-gateway/src/routes/messages/mod.rs`
  that routes a Bedrock response through a normalizing SSE stream
- `bedrock_messages_harness.sh`, a one-command live check against real Bedrock
- failing tests that specify the behavior you need to produce

Two facts about Bedrock that would otherwise cost you an hour of reading:

- non-streaming requests go to `POST /model/{modelId}/invoke`; streaming goes
  to `POST /model/{modelId}/invoke-with-response-stream`. The model id is in
  the path, not the body
- the streaming response is `application/vnd.amazon.eventstream`, a binary
  framing format, not SSE. Each frame's payload is
  `{"bytes": "<base64 of one Anthropic event as JSON>"}`.
  `aws_smithy_eventstream::frame::MessageFrameDecoder` parses the framing for
  you and is already a dependency

## What is yours to write

Everything marked `todo!()`, plus whatever the compiler and the failing tests
tell you is missing:

- `crates/core/src/providers/bedrock/messages/transformation.rs`, the Bedrock
  provider config: URL construction, region resolution, the request transform,
  and the response transform. `providers/anthropic/messages/transformation.rs`
  and `providers/azure_ai/messages/transformation.rs` are the two existing
  implementations of the same trait; read them first
- whatever the shared contract in `crates/core/src/messages/transformation.rs`
  needs in order to express Bedrock. It currently cannot express everything
  Bedrock requires, and part of this task is deciding how to extend it
- the EventStream decode loop in `crates/ai-gateway/src/routes/messages/mod.rs`

## Running it

```bash
cd litellm-rust
cargo run -p litellm-ai-gateway --features server   # BEDROCK_MODEL, PORT, LITELLM_MASTER_KEY from env
./bedrock_messages_harness.sh                       # in another shell
```

The harness runs four scenarios: a non-streaming turn, a streaming turn, a
tool_use round trip, and an invalid model id

## Definition of done

1. all four harness scenarios behave correctly against live Bedrock
2. the Bedrock tests in core and in the messages route pass
3. `cargo test --workspace` is green, including the roughly 150 tests that
   existed before you started
4. `cargo fmt --check` and
   `cargo clippy -p litellm-ai-gateway --all-targets --features server -- -D warnings`
   are clean
5. Claude Code, pointed at the gateway, completes a turn

To point Claude Code at the gateway, set `ANTHROPIC_BASE_URL` to the gateway's
address, `ANTHROPIC_AUTH_TOKEN` to the gateway master key, and `ANTHROPIC_MODEL`
to the Bedrock model id, with `ANTHROPIC_API_KEY` unset

## Not in scope

`POST /v1/messages/count_tokens`, multi-deployment routing, and Python-side
integration. If you find yourself in any of those, come up for air

Read `CLAUDE.md` and `AGENTS.md` in this directory before you start. They are
the standards this code is held to
