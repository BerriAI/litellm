import { describe, expect, it } from "vitest";

import {
  autoRoutersOf,
  buildCachePayload,
  buildCompressionGuardrailPayload,
  buildComplexityAutorouterPayload,
  compressionGuardrailsOf,
} from "./helpers";

describe("compressionGuardrailsOf", () => {
  it("keeps only headroom-provider guardrails and drops others", () => {
    const filtered = compressionGuardrailsOf({
      guardrails: [
        { guardrail_id: "1", guardrail_name: "headroom-compression", litellm_params: { guardrail: "headroom" } },
        { guardrail_id: "2", guardrail_name: "pii-masker", litellm_params: { guardrail: "presidio" } },
        { guardrail_id: "3", guardrail_name: "no-params", litellm_params: null },
      ],
    });

    expect(filtered.map((g) => g.guardrail_id)).toEqual(["1"]);
  });
});

describe("buildCompressionGuardrailPayload", () => {
  it("builds a headroom guardrail payload with trimmed fields", () => {
    const payload = buildCompressionGuardrailPayload({
      name: "  headroom-compression  ",
      apiBase: "  https://compress  ",
      defaultOn: false,
    });

    expect(payload).toEqual({
      guardrail_name: "headroom-compression",
      litellm_params: {
        guardrail: "headroom",
        mode: "pre_call",
        api_base: "https://compress",
        default_on: false,
      },
    });
  });
});

describe("buildCachePayload", () => {
  it("coerces the port to a number and omits an empty password", () => {
    expect(buildCachePayload({ host: " my-host ", port: "6379", password: "  " })).toEqual({
      type: "redis",
      host: "my-host",
      port: 6379,
    });
  });

  it("includes a trimmed password when provided", () => {
    const expected = {
      type: "redis",
      host: "h",
      port: 6380,
      password: "secret",
    };
    expect(buildCachePayload({ host: "h", port: "6380", password: " secret " })).toEqual(expected);
  });
});

describe("autoRoutersOf", () => {
  it("keeps only auto_router deployments", () => {
    const filtered = autoRoutersOf({
      data: [
        { model_name: "cost-router", litellm_params: { model: "auto_router/complexity_router" } },
        { model_name: "gpt-4", litellm_params: { model: "openai/gpt-4" } },
      ],
    });

    expect(filtered.map((d) => d.model_name)).toEqual(["cost-router"]);
  });
});

describe("buildComplexityAutorouterPayload", () => {
  it("wraps tiers and default model into a complexity_router deployment", () => {
    const payload = buildComplexityAutorouterPayload({
      name: " cost-router ",
      defaultModel: "gpt-4o-mini",
      tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: [], COMPLEX: ["gpt-4o"], REASONING: [] },
    });

    expect(payload).toEqual({
      model_name: "cost-router",
      litellm_params: {
        model: "auto_router/complexity_router",
        complexity_router_config: {
          tiers: { SIMPLE: ["gpt-4o-mini"], MEDIUM: [], COMPLEX: ["gpt-4o"], REASONING: [] },
          classifier_type: "heuristic",
        },
        complexity_router_default_model: "gpt-4o-mini",
      },
      model_info: {},
    });
  });
});
