# Fusion models

Fusion models are virtual LiteLLM model groups that give one **outer model** a private deliberation tool. They remain compatible with normal chat, Responses API, Anthropic Messages, streaming, and client tool loops.

Deliberation is an optional private tool inside an otherwise normal model call, rather than an always-on panel in front of every request.

The request path is:

1. LiteLLM adds a private `litellm_fusion` server tool alongside any client tools and calls the outer model normally.
2. If requested, 1–8 panel models answer a self-contained question in parallel.
3. The analyst compares consensus, contradictions, partial coverage, unique insights, and blind spots. It does not choose a winner or write the final response.
4. The outer model receives the structured analysis and bounded raw responses, then returns the only client-visible answer or tool call.

If the outer model answers directly or selects a client tool, LiteLLM returns that first response without running a second outer-model completion. Panel and analyst models never receive client tools. If they use an optional LiteLLM Search Tool, the search is executed server-side and its results remain advisory. A failed panel is reported to the outer model; one successful panel is enough to continue. If the analyst fails or returns invalid JSON, the outer model still receives the raw panel responses. If every panel fails, the outer model receives a typed error and can answer without them.

## Configuration

```yaml
model_list:
  # These are user-defined model-group names for regular deployments that
  # already exist on the proxy.
  - model_name: fusion/general
    litellm_params:
      model: fusion_router
      fusion_router_config:
        outer_model: production-outer
        panel_models: [research-fast, research-reasoning]
        analyst_model: production-outer # optional; defaults to outer_model
        invocation: auto # auto or required
        reasoning_effort: none
        temperature: 0
        max_completion_tokens: 16000
        panel_timeout_seconds: 120
        max_candidate_chars: 12000
        # Optional existing LiteLLM Search Tool:
        # search_tool_name: web-search
        # max_tool_calls: 4
```

Call `fusion/general` exactly like any other model. `invocation: auto` lets the outer model skip the panel for routine requests. `required` forces deliberation and is useful for evaluations or workloads where every request should receive the same treatment.

`reasoning_effort: none` makes deliberation replace private extended reasoning where a provider supports that parameter. LiteLLM drops it for providers that do not support it. The optional Search Tool supplies search results and bounded page content through LiteLLM's Search API. This first version does not expose a separate URL-fetch tool.

The outer model must support function calling. Panel and analyst models only need function calling when a Search Tool is configured. Granting access to the Fusion model lets the request use its administrator-configured model and search dependencies; the panel query and private research are sent to those deployments under their normal provider data policies.

## Operational behavior

- The outer model is the only hard health dependency. Panel failures degrade into tool-result data, and analyst failure degrades to raw responses.
- Initial outer, panel, analyst, continuation, and search calls are marked separately in spend logs. They inherit the caller identity and remain part of one logical Fusion request.
- Admission control reserves the worst-case model-call cost. Hidden calls accumulate against that shared reservation, and the direct initial response or final continuation reconciles it once. This keeps concurrent requests from spending the same remaining budget while Fusion is still running.
- Chat-completion streaming is buffered until LiteLLM knows whether the private tool was invoked. A direct response is replayed as a normal stream; a Fusion invocation suppresses the private tool-call stream and exposes only the final outer-model stream.
- A request-level `tool_choice: required` is considered satisfied when Fusion runs. The continuation changes it to `auto` when client tools exist, or removes it when they do not, so the outer model can finish instead of being forced into a second tool call.
- A client tool named `litellm_fusion` is rejected because that name is reserved for the private server tool.
- A Fusion model cannot use another Fusion model as its outer, panel, or analyst model. The router's existing recursion guard enforces this at runtime.
- Fusion runs at most once per top-level model request. The harness still owns the multi-turn tool loop, so a later tool result creates a new model request and a new independent Fusion decision.
