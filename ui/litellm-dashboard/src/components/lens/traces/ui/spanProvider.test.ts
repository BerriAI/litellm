import { describe, expect, it } from "vitest";

import { costMapLookup, resolveSpanProvider } from "./spanProvider";

const costMap = {
  "gpt-6.1-sol": { litellm_provider: "openai" },
  "claude-sonnet-5-5": { litellm_provider: "anthropic" },
  "no-provider-model": { input_cost_per_token: 1 },
};
const lookup = costMapLookup(costMap);

describe("resolveSpanProvider", () => {
  it("uses a known provider prefix without consulting the cost map", () => {
    expect(resolveSpanProvider("anthropic/unlisted-model", costMapLookup({}))).toBe("anthropic");
  });

  it("looks the model up in the cost map", () => {
    expect(resolveSpanProvider("gpt-6.1-sol", lookup)).toBe("openai");
  });

  it("falls back to the bare model name when the prefix is not a provider", () => {
    expect(resolveSpanProvider("my-team/claude-sonnet-5-5", lookup)).toBe("anthropic");
  });

  it("returns null for unknown models, missing providers and empty input", () => {
    expect(resolveSpanProvider("totally-unknown", lookup)).toBeNull();
    expect(resolveSpanProvider("no-provider-model", lookup)).toBeNull();
    expect(resolveSpanProvider(null, lookup)).toBeNull();
  });

  it("tolerates a cost map that is not an object", () => {
    expect(resolveSpanProvider("gpt-6.1-sol", costMapLookup(undefined))).toBeNull();
    expect(resolveSpanProvider("gpt-6.1-sol", costMapLookup("oops"))).toBeNull();
  });
});
