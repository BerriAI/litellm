# Task: serve AWS Bedrock on `POST /v1/messages`

The gateway in this workspace already serves the Anthropic Messages API for
Anthropic and Azure. Your job is to make it serve AWS Bedrock too

The bar is not "the tests go green". The bar is that we finish the day
confident that real Claude Code traffic can run through the Rust gateway
against a Bedrock-hosted Claude, and that nothing that worked before is
broken. LiteLLM's Python implementation of this endpoint is the thing Rust is
replacing: the Python entry point should stop doing the work itself and
delegate to your Rust code instead, with the same behavior on the other side

Claude Code always sends `stream: true`, so streaming is part of the job, not a
stretch goal

## Start here

The starter branch is
[`litellm_bedrock_messages_interview_starter`](https://github.com/ishaan-berri/litellm/tree/litellm_bedrock_messages_interview_starter).
It has the scaffolding in place, the placeholders you will fill in, and the
tests that specify the behavior

```bash
git clone https://github.com/ishaan-berri/litellm.git
cd litellm
git checkout litellm_bedrock_messages_interview_starter
make bootstrap                      # provisions everything the tests and the proxy need
```

Two builds matter, and they are separate. `cargo build` inside `litellm-rust`
compiles the gateway. The Python bridge is a native extension that has to be
rebuilt and reinstalled every time you change Rust, or Python will keep running
the last build you made and your edits will appear to do nothing:

```bash
uv run --with maturin==1.9.4 maturin develop \
  --manifest-path litellm-rust/crates/python-bridge/Cargo.toml   # from the repo root
```

Read `CLAUDE.md` and `AGENTS.md` in this directory before you start. They are
the standards this code is held to

## What you are given

Credentials and target, already in your environment:

- `AWS_BEARER_TOKEN_BEDROCK`, a Bedrock API key. Bedrock accepts it as a plain
  `Authorization: Bearer <token>` header, so you do not need to implement SigV4
- Region `us-west-2`, model `us.anthropic.claude-sonnet-4-5-20250929-v1:0`.
  The Claude 3.5 and 3.7 Bedrock ids are end of life and return 404

Unset `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` in any shell you test
from. If they are present, the code takes the SigV4 path instead of the bearer
path and you will chase a 403 that has nothing to do with your change

Already wired up, so you should not need to touch it:

- provider registration, the `BEDROCK_MODEL` startup config, the bearer and
  SigV4 auth paths in `crates/ai-gateway/src/messages/{prepare,handler}.rs`,
  and the streaming plumbing in `crates/ai-gateway/src/routes/messages/mod.rs`
  that routes a Bedrock response through a normalizing SSE stream
- `bedrock_messages_harness.sh`, a one-command live check against real Bedrock
- failing tests that specify the behavior you need to produce

## Talk to Bedrock directly first

Before touching Rust, spend two minutes confirming what the upstream API
actually does. The model id lives in the URL path, not the body, and the two
verbs differ only by the last path segment

Non-streaming:

```bash
export MODEL_ID='us.anthropic.claude-sonnet-4-5-20250929-v1:0'

curl -sS -X POST \
  "https://bedrock-runtime.us-west-2.amazonaws.com/model/${MODEL_ID}/invoke" \
  -H "Authorization: Bearer ${AWS_BEARER_TOKEN_BEDROCK}" \
  -H 'Content-Type: application/json' \
  -d '{
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "Say hello in five words"}]
      }'
```

returns a plain Anthropic Messages response, with no `model` field in the
request body and `anthropic_version` in its place:

```json
{"model":"claude-sonnet-4-5-20250929","id":"msg_bdrk_01UoJmSqSyXtTHfSNA7kWg3B","type":"message","role":"assistant","content":[{"type":"text","text":"Hello, how are you doing?"}],"stop_reason":"end_turn","usage":{"input_tokens":12,"output_tokens":10}}
```

Streaming, same body, different path segment:

```bash
curl -sS -X POST \
  "https://bedrock-runtime.us-west-2.amazonaws.com/model/${MODEL_ID}/invoke-with-response-stream" \
  -H "Authorization: Bearer ${AWS_BEARER_TOKEN_BEDROCK}" \
  -H 'Content-Type: application/json' \
  -d '{
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 32,
        "messages": [{"role": "user", "content": "Say hello"}]
      }' | xxd | head
```

returns `application/vnd.amazon.eventstream`, a binary framing format rather
than SSE. Piping it through `xxd` shows the shape you have to decode:

```
00000000: 0000 02b5 0000 004b 11c3 611e 0b3a 6576  .......K..a..:ev
00000010: 656e 742d 7479 7065 0700 0563 6875 6e6b  ent-type...chunk
00000050: 0005 6576 656e 747b 2262 7974 6573 223a  ..event{"bytes":
00000060: 2265 794a 3065 5842 6c49 6a6f 6962 5756  "eyJ0eXBlIjoibWV
```

Each frame's payload is `{"bytes": "<base64 of one Anthropic event as JSON>"}`.
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

The Rust side, gateway plus a live check against real Bedrock:

```bash
cd litellm-rust
cargo run -p litellm-ai-gateway --features server   # BEDROCK_MODEL, PORT, LITELLM_MASTER_KEY from env
./bedrock_messages_harness.sh                       # in another shell
```

The harness runs four scenarios: a non-streaming turn, a streaming turn, a
tool_use round trip, and an invalid model id

The Rust unit tests, which are the written spec for the transformation:

```bash
cd litellm-rust
cargo test -p litellm-core --features bedrock-auth
```

On the starter branch that reports 123 passed and 5 failed, every failure a
`not yet implemented` panic from one of the placeholders you are filling in

## The Python side

This is the half the old version of this brief left out, and it is where
"Rust replaces Python" actually gets proven

`litellm.anthropic.messages.acreate(..., rust=True)` routes the call through
the native bridge in `litellm/rust_bridge/` instead of the Python
implementation, and stamps `x-litellm-rust: true` on the response so a caller
can tell which engine served it. The live test for that path is
`tests/test_litellm/anthropic_interface/test_bedrock_rust_bridge_e2e.py`, one
non-streaming case and one streaming case against real Bedrock:

```bash
unset AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY
export AWS_BEARER_TOKEN_BEDROCK='...'
uv run pytest tests/test_litellm/anthropic_interface/test_bedrock_rust_bridge_e2e.py -v
```

Both tests skip when the token is absent, so a green run with no token means
nothing ran. On the starter branch, with a token, they fail on a missing
`x-litellm-rust` key. Read that failure carefully, because it is the trap in
this task: when your Rust code panics on a `todo!()`, the bridge catches it and
silently falls back to the Python implementation. The request succeeds, you get
a real answer from Bedrock, and the only evidence that Rust never ran is the
missing header. Any time a test passes here, confirm it passed through Rust

The mocked companion,
`tests/test_litellm/anthropic_interface/test_rust_bridge_messages.py`, covers
the gate itself (the `rust=True` and `LITELLM_RUST` switches, streaming
forwarding, the header, and the fallback) and needs no credentials

Beyond the SDK entry point, we have Python end-to-end suites that drive
`POST /v1/messages` on a live proxy. You are not expected to make all of these
pass today, but read them: they are the regression bar this work is eventually
measured against, and they will tell you what real Claude Code traffic looks
like

- `tests/e2e/claude_code/` is a feature x provider matrix that drives the
  actual `claude` CLI against a running proxy. The Bedrock Invoke column is the
  one you care about, for example
  `tests/e2e/claude_code/basic_messaging_non_streaming/test_bedrock_invoke.py`
  and `.../basic_messaging_streaming/test_bedrock_invoke.py`, with further rows
  for tool use, thinking, structured outputs, prompt caching, and token counting
- `tests/e2e/llm_translation/test_messages_e2e.py` covers `/v1/messages`
  non-streaming, streaming, tool use, and cost logging against Anthropic
- `tests/e2e/llm_translation/test_messages_azure_foundry_e2e.py` already
  asserts the `x-litellm-rust` header when `E2E_EXPECT_RUST=1`, which is the
  pattern for proving a route has moved to Rust
- `tests/e2e/CONTRIBUTING.md` explains how to bring up the proxy these need

## Definition of done

Success is a Claude Code session running against Bedrock Invoke through the
Rust gateway, with evidence that Rust served it and evidence that nothing else
regressed. Concretely:

1. Claude Code, pointed at the gateway, completes a turn against Bedrock. Set
   `ANTHROPIC_BASE_URL` to the gateway's address, `ANTHROPIC_AUTH_TOKEN` to the
   gateway master key, and `ANTHROPIC_MODEL` to the Bedrock model id, with
   `ANTHROPIC_API_KEY` unset
2. all four `bedrock_messages_harness.sh` scenarios behave correctly against
   live Bedrock
3. both tests in `test_bedrock_rust_bridge_e2e.py` pass, and pass *through
   Rust*: the `x-litellm-rust` assertion holds, so the Python implementation is
   no longer doing the work
4. the Bedrock tests in core and in the messages route pass
5. `cargo test --workspace` is green, including the roughly 150 tests that
   existed before you started, and
   `uv run pytest tests/test_litellm/anthropic_interface/ -v` is green
6. `cargo fmt --check` and
   `cargo clippy -p litellm-ai-gateway --all-targets --features server -- -D warnings`
   are clean

Zero regressions is a hard requirement, not a nice-to-have. If you change the
shared trait in `crates/core/src/messages/transformation.rs` to fit Bedrock,
Anthropic and Azure ride on that same trait, and you own proving they still
work

We will spend the last part of the day on how confident you are and why. Come
prepared to say which behaviors you proved, which you inferred, and where you
would still expect this to break in production

## Not in scope

`POST /v1/messages/count_tokens`, multi-deployment routing, and migrating any
endpoint other than Messages. The Python work is limited to making the existing
bridge tests pass through Rust; you do not need to change the proxy, the
router, or the Python Bedrock implementation itself. If you find yourself in
any of those, come up for air
