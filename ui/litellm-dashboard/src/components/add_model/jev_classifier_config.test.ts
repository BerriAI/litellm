import { describe, expect, it } from "vitest";
import {
  jevClassifierConfigSchema,
  jevClassifierFormConfigSchema,
  normalizeJevClassifierConfig,
  transitionDecisionModelProvider,
} from "./jev_classifier_config";

const nimble = { provider: "bespoke_nimble" as const, model: "nimble-latest", timeout_ms: 3000 };

describe("decision model connections", () => {
  it("keeps new connection overrides but strips saved credentials on hydration", () => {
    const supplied = { ...nimble, api_base: " https://nimble.example.com ", api_key: " new-key " };
    expect(normalizeJevClassifierConfig(jevClassifierFormConfigSchema.parse(supplied))).toEqual({
      ...nimble,
      api_base: "https://nimble.example.com",
      api_key: "new-key",
    });
    expect(jevClassifierConfigSchema.parse(supplied)).toEqual(nimble);
  });

  it.each([
    [{ api_base: "", api_key: "  " }, {}],
    [{ api_key: null }, { api_key: null }],
    [
      { api_base: null, api_key: null },
      { api_base: null, api_key: null },
    ],
  ])("distinguishes untouched fields from explicit connection resets: %j", (input, expected) => {
    expect(normalizeJevClassifierConfig({ ...nimble, ...input })).toEqual({ ...nimble, ...expected });
  });

  it("drops credentials when changing providers and keeps tuning settings", () => {
    const jev = { model: "custom-jev", timeout_ms: 9000, api_base: "https://jev.example.com", api_key: "jev-key" };
    const switched = transitionDecisionModelProvider(jev, "bespoke_nimble");
    expect(switched).toEqual({ ...nimble, timeout_ms: 9000 });
    expect(transitionDecisionModelProvider({ ...switched, api_key: "nimble-key" }, "typesafe")).toEqual({
      provider: "typesafe",
      model: "jev-latest",
      timeout_ms: 9000,
    });
    expect(transitionDecisionModelProvider(jev, "typesafe")).toBe(jev);
  });
});
