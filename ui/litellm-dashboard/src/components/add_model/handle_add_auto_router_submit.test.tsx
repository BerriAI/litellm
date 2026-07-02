import { describe, expect, it } from "vitest";
import { prepareAutoRouterTestDeployment, resolveAutoRouterDefaultModel } from "./handle_add_auto_router_submit";

const tiers = (overrides: Partial<Record<"SIMPLE" | "MEDIUM" | "COMPLEX" | "REASONING", string>> = {}) => ({
  SIMPLE: "",
  MEDIUM: "",
  COMPLEX: "",
  REASONING: "",
  ...overrides,
});

describe("resolveAutoRouterDefaultModel", () => {
  it("prefers the MEDIUM tier for a complexity router", () => {
    expect(resolveAutoRouterDefaultModel("complexity", tiers({ SIMPLE: "small", MEDIUM: "mid", COMPLEX: "big" }))).toBe(
      "mid",
    );
  });

  it("falls back through the tiers when MEDIUM is empty", () => {
    expect(resolveAutoRouterDefaultModel("complexity", tiers({ COMPLEX: "big", REASONING: "reason" }))).toBe("big");
  });

  it("returns an empty string when no complexity tier is set", () => {
    expect(resolveAutoRouterDefaultModel("complexity", tiers())).toBe("");
  });

  it("uses the explicit default model for a semantic router", () => {
    expect(resolveAutoRouterDefaultModel("semantic", tiers({ MEDIUM: "ignored" }), "gpt-4o")).toBe("gpt-4o");
  });
});

describe("prepareAutoRouterTestDeployment", () => {
  it("builds a single deployment that targets the default model", () => {
    expect(prepareAutoRouterTestDeployment("gpt-4o-mini")).toEqual([
      { litellmParamsObj: { model: "gpt-4o-mini" }, modelInfoObj: {}, modelName: "gpt-4o-mini" },
    ]);
  });

  it("returns no deployments when there is no default model", () => {
    expect(prepareAutoRouterTestDeployment("")).toEqual([]);
  });
});
