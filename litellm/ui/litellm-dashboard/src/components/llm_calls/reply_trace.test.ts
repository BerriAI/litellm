import { describe, expect, it } from "vitest";
import { gatewayTraceFromHeaders } from "./reply_trace";

describe("gatewayTraceFromHeaders", () => {
  it("reads routing, timing, cost and guardrail details from the gateway headers", () => {
    const headers = new Headers([
      ["x-litellm-call-id", "call-1"],
      ["x-litellm-model-group", "fast"],
      ["x-litellm-model-id", "deployment-1"],
      ["x-litellm-model-name", "provider/model-a"],
      ["x-litellm-model-api-base", "https://provider.test/v1"],
      ["x-litellm-attempted-fallbacks", "1"],
      ["x-litellm-attempted-retries", "2"],
      ["x-litellm-overhead-duration-ms", "12.5"],
      ["x-litellm-response-duration-ms", "840"],
      ["x-litellm-response-cost", "0.0004"],
      ["x-litellm-applied-guardrails", "pii-mask, toxicity"],
      ["x-litellm-applied-policies", "strict"],
      ["x-litellm-cache-key", "abc"],
    ]);
    const trace = gatewayTraceFromHeaders(headers);

    const expected = {
      callId: "call-1",
      modelGroup: "fast",
      deploymentId: "deployment-1",
      providerModel: "provider/model-a",
      apiBase: "https://provider.test/v1",
      attemptedFallbacks: 1,
      attemptedRetries: 2,
      overheadMs: 12.5,
      responseMs: 840,
      cost: 0.0004,
      appliedGuardrails: ["pii-mask", "toxicity"],
      appliedPolicies: ["strict"],
      cacheHit: true,
    };
    expect(trace).toEqual(expected);
  });

  it("leaves details unset when headers are missing, blank or not numeric", () => {
    const trace = gatewayTraceFromHeaders(
      new Headers({
        "x-litellm-model-group": "  ",
        "x-litellm-attempted-fallbacks": "none",
        "x-litellm-applied-guardrails": " , ",
      }),
    );

    expect(trace.modelGroup).toBeUndefined();
    expect(trace.attemptedFallbacks).toBeUndefined();
    expect(trace.appliedGuardrails).toEqual([]);
    expect(trace.cacheHit).toBe(false);
  });

  it("leaves policies unknown when the gateway's own headers are hidden from the page", () => {
    const hidden = gatewayTraceFromHeaders(new Headers({ "x-litellm-applied-guardrails": "pii-mask" }));
    const visible = gatewayTraceFromHeaders(new Headers({ "x-litellm-call-id": "call-1" }));

    expect(hidden.appliedGuardrails).toEqual(["pii-mask"]);
    expect(hidden.appliedPolicies).toBeUndefined();
    expect(visible.appliedPolicies).toEqual([]);
  });
});
