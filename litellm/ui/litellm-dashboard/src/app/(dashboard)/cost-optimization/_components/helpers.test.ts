import { describe, expect, it } from "vitest";

import { buildCompressionGuardrailPayload, compressionGuardrailsOf } from "./helpers";

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
