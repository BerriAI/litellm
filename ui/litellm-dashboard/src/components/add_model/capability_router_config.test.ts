import { describe, expect, it } from "vitest";

import {
  capabilityRouterConfigError,
  defaultCapabilityRouterConfig,
  hydrateCapabilityRouterConfig,
} from "./capability_router_config";

const validConfig = () => ({
  ...defaultCapabilityRouterConfig(),
  candidates: [
    { model: "small", description: "Good for bounded extraction" },
    { model: "frontier", description: "Good for ambiguous multi-step work" },
  ],
  classifier: { model: "classifier", timeout_ms: 3000, max_output_tokens: 1024 },
  fallback_model: "frontier",
});

describe("capability router config", () => {
  it("accepts the minimal complete configuration", () => {
    expect(capabilityRouterConfigError(validConfig())).toBeNull();
  });

  it("requires descriptions, unique models, and a candidate fallback", () => {
    expect(
      capabilityRouterConfigError({
        ...validConfig(),
        candidates: [{ model: "small", description: "" }, validConfig().candidates[1]],
      }),
    ).toContain("Describe");
    expect(
      capabilityRouterConfigError({
        ...validConfig(),
        candidates: [validConfig().candidates[0], validConfig().candidates[0]],
      }),
    ).toContain("unique");
    expect(capabilityRouterConfigError({ ...validConfig(), fallback_model: "other" })).toContain("fallback");
  });

  it("hydrates stored values while supplying newly introduced defaults", () => {
    const hydrated = hydrateCapabilityRouterConfig({
      candidates: validConfig().candidates,
      classifier: { model: "classifier" },
      fallback_model: "frontier",
      probability_threshold: 0.8,
    });

    expect(hydrated.probability_threshold).toBe(0.8);
    expect(hydrated.classifier.timeout_ms).toBe(3000);
    expect(hydrated.cache_ttl_seconds).toBe(3600);
  });
});
