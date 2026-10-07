# structure

These examples cover the complete Messages stack. Some provider implementations and test ports land after the shared contracts; inspect the current branch before assuming support

A Python test port follows the source setup, assertions and parameterized scenarios through the Rust adapter's actual inputs and caller. Map behavior rather than helper names or collection counts

Test private helpers inline at the owning file's end. Public-only tests belong in the owning crate's `tests/<subject>.rs`. Use named `rstest` cases and typed injected fixtures. Keep a fixture focused on setup and assertions about one result together

Boundary projections of Python globals, cost-map capabilities and kwargs belong in `tests/unit/rust_bridge/<format>`. Provider rewrites belong in Rust adapter tests; route execution belongs in the relevant inference crate or recording-server integration harness

# examples

## A port exposes missing implementation

Replacing `Error::Unsupported` with an implementation is not inherently a scope violation. Determine whether the implementation is authorized and whether it stays within the source contract. A request to port tests with simple fixes and report larger failures does not become unrestricted parity work

When the scope is `litellm-rust/` only, do not change Python implementations, fixtures or guides to make a port pass. Shared wire machinery and its tests land in the foundation PR; a provider example changes only its adapter, inline tests, guide and module export. Check each PR diff independently for these boundaries

For a simple failure, retain the source assertion, fix the behavior and run the affected case. Examples include preserving thinking display during enabled-to-adaptive conversion or translating an Azure `api-key` to `x-api-key`

For an unresolved dependency outside the authorized scope, report the input, expected/actual result and required implementation. Do not weaken assertions, ignore the test or count an unresolved scenario as ported

```json
{
  "source": "test_source_behavior",
  "status": "implementation_required",
  "expected": "provider wire behavior from the source assertion",
  "actual": "native behavior demonstrated by the failing case",
  "required_work": "specific missing transformation or integration"
}
```

This is a report example, not a mandated new ledger schema. Use the existing ledger's fields and conventions when one exists

## Preserve semantic assertions

A schema regression should assert both normalization and preservation rather than merely `is_ok()`

```rust
#[rstest]
#[case::constant("const", json!({"type": "custom"}))]
#[case::examples("examples", json!([{"type": "custom"}]))]
fn literals_survive(#[case] field: &str, #[case] literal: Value) {
    let input = json!({
        "name": "example",
        "input_schema": {"type": "custom", field: literal.clone()}
    });
    let tool = serde_json::from_value::<ToolDefinition>(input).unwrap();
    let output = serde_json::to_value(bedrock_tool_definition(tool, 0, false)).unwrap();
    assert_eq!(output, json!({
        "name": "example",
        "input_schema": {"type": "object", field: literal}
    }));
}
```

The owning module supplies the imports. This test fails if normalization is removed or if literals are rewritten

## Separate parity, compatibility and scope

Consider Vertex authentication: the source test requires a stale forwarded bearer not to suppress credential refresh. Keeping arbitrary forwarded credentials would fail that source contract. Removing them also changes the earlier Rust behavior

Report both conclusions: the implementation matches the source test, and forwarded-bearer-only callers need configured credentials. A behavior change is not automatically a defect; a passing port is not automatically compatibility evidence

Similarly, native Messages system normalization follows the existing Python helper. Reverting it solely because older Rust assertions changed would restore a parity gap

## Trace the complete path

The Bedrock schema helper is consumed by the actual tool transform, region projection feeds URL/signing preparation, and the wire hook feeds the handler. Test those observable connections where a disconnected helper could otherwise pass

Workspace forwarding on Invoke is an additional behavior with Mantle-only reference coverage. Explicit AWS credential projection is a pre-existing gap. Distinguish these from the confirmed literal-rewrite and region-validation bugs

The Rust-only stack leaves Python's Bedrock kwargs and capability projection unchanged. Region, endpoint, workspace and metadata settings therefore need a separate host-integration follow-up. A historical ledger may include tests proposed alongside Python fixes; label that snapshot and never use it as evidence of parity with unchanged Python

Decoder tests plus a shared HTTP-error mapping preserve upstream status. A public Chat helper using that decoder does not prove deployed Chat reachability. Check provider selection and route calls before claiming blast radius

# validation

Run from the repository root, narrowing the filter to affected cases before broadening to their provider suite

```sh
cargo test --manifest-path litellm-rust/Cargo.toml --locked -p litellm-llms --lib bedrock::messages
cargo test --manifest-path litellm-rust/Cargo.toml --locked -p litellm-inference-messages
.venv/bin/pytest -p no:rerunfailures tests/unit/rust_bridge/messages/test_route_host.py -q
```

The rerun plugin is disabled here because it opens a socket in these in-process unit tests. Record the result and whether the extension was rebuilt; Python projection tests alone do not verify installed native integration

In an existing case ledger, keep source mappings and literal expectations meaningful. Source definitions, expanded scenarios, existing coverage, added regressions and exclusions are different counts. Logs referring to an earlier revision or another working tree do not verify the present checkout

Preserve malformed-input acceptance intentionally when translating dynamic Python fixtures into typed Rust data. Document a rejected source fixture as a divergence or exclusion; never quietly remove it and claim equivalent coverage

# references

- `litellm-rust/AGENTS.md`
- `tests/unit/AGENTS.md`
- `tests/unit/rust_bridge/AGENTS.md`
- `litellm-rust/messages-unit-test-port.json`
- `tests/unit/llms/bedrock/test_request_metadata.py`
- `tests/unit/llms/bedrock/test_bedrock_common_utils.py`
