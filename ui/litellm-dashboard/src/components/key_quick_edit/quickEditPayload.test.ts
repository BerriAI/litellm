import { describe, expect, it } from "vitest";
import { budgetQuickEditPayload, modelsQuickEditPayload } from "./quickEditPayload";

describe("budgetQuickEditPayload", () => {
  it("maps an empty budget to unlimited and preserves the reset period", () => {
    expect(budgetQuickEditPayload("key-token", { maxBudget: "", budgetDuration: "30d" })).toEqual({
      kind: "ok",
      payload: { key: "key-token", max_budget: null, budget_duration: "30d" },
    });
  });

  it("parses a finite non-negative budget", () => {
    expect(budgetQuickEditPayload("key-token", { maxBudget: "12.5", budgetDuration: null })).toEqual({
      kind: "ok",
      payload: { key: "key-token", max_budget: 12.5, budget_duration: null },
    });
  });

  it.each(["-1", "NaN", "Infinity"])("rejects invalid budget input %s", (maxBudget) => {
    expect(budgetQuickEditPayload("key-token", { maxBudget, budgetDuration: null })).toEqual({
      kind: "invalid",
      message: "Max budget must be a finite, non-negative number.",
    });
  });
});

describe("modelsQuickEditPayload", () => {
  it("collapses the all-team sentinel", () => {
    expect(modelsQuickEditPayload("key-token", ["gpt-4o-mini", "all-team-models"])).toEqual({
      key: "key-token",
      models: ["all-team-models"],
    });
  });

  it("collapses the all-proxy sentinel when no team sentinel is selected", () => {
    expect(modelsQuickEditPayload("key-token", ["gpt-4o-mini", "all-proxy-models"])).toEqual({
      key: "key-token",
      models: ["all-proxy-models"],
    });
  });

  it("preserves ordinary model selections", () => {
    expect(modelsQuickEditPayload("key-token", ["gpt-4o-mini", "gpt-3.5-turbo"])).toEqual({
      key: "key-token",
      models: ["gpt-4o-mini", "gpt-3.5-turbo"],
    });
  });
});
