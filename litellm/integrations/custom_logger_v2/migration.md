# Migrating custom callbacks to Rust

Inventory the callbacks that apply to each proxy route, model and policy before moving any code. Include callbacks under `litellm_settings.callbacks`, configured guardrails and callbacks that run only on streaming requests. Record what each reads and writes, whether it can reject a request, and which later callbacks depend on its output.

The Python proxy can use Rust components such as the router while it continues to run custom Python callbacks. The [native gateway roadmap](https://docs.litellm.ai/rust-migration) describes forwarding requests that need unsupported callbacks to a Python worker. Measure the route and engine your request actually uses when comparing latencies.

## Move one callback at a time

1. Capture the callback's behavior using synthetic requests. Include requests that should pass, requests that should be rejected, any request mutations, and streaming output if applicable.
2. Profile the callback on the Python proxy. If local computation dominates, you can move that function into an extension you own and call it from the existing Python callback. Keep app configuration and callback registration as they are for this step.
3. Repeat the same tests and compare pre-call latency with and without the extension. Include data conversion and any thread scheduling in the measurement. If your callback makes network calls, measure their time separately.
4. For a native gateway port, first check which hooks, fields and outcomes the target LiteLLM version exposes. Follow [the proposed native callback contract](https://github.com/BerriAI/litellm/pull/51526) for work in progress. Verify registration and request eligibility on the gateway before expecting the full request to run in Rust.

For a callback that blocks outgoing content, include a test in which an earlier callback adds that content. Assert both that the request is rejected and that the provider receives zero calls. Keep the Python path available until the native path covers all the callbacks required by that request.

## Use the built-in content filter as a parity example

Add the following to a test proxy config with a model named `demo` pointed at a local mock provider. Bind the proxy to localhost and use only synthetic inputs.

```yaml
guardrails:
  - guardrail_name: demo-code-block
    litellm_params:
      guardrail: litellm_content_filter
      mode: pre_call
      default_on: true
      patterns:
        - pattern_type: regex
          name: demo_code
          pattern: '\bDEMO-[0-9]{6}\b'
          action: BLOCK
```

Send two requests to `/v1/chat/completions`, using the configured proxy key if required. These examples reuse the same endpoint and provider so the request content is the only difference.

```sh
curl -i http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"demo","messages":[{"role":"user","content":"demo-123456"}]}'

curl -i http://localhost:4000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"demo","messages":[{"role":"user","content":"demo-12345"}]}'
```

The first request should be blocked before the mock receives it. It also checks case-insensitive matching. The second should reach the mock once. Use the same configuration for a candidate port and compare both results, including the denial response and provider call counts.

A first port could move just the regex matcher behind the existing filter. Before expanding coverage, check Python regex syntax and Unicode behavior, pattern order, and offsets used for masking. Keep a Python path for patterns whose semantics the Rust implementation cannot preserve. A full native filter also needs to preserve which fields and turns are inspected, how matches are recorded, and how streamed content is handled.
