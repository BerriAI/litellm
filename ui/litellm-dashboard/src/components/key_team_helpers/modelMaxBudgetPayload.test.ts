import { describe, expect, it } from "vitest";
import { modelMaxBudgetUpdate } from "./modelMaxBudgetPayload";

const GPT_4O = { "gpt-4o": { budget_limit: 5, time_period: "30d" } };

describe("modelMaxBudgetUpdate", () => {
  it("sends the edited budgets when a row is added to a key that had none", () => {
    expect(modelMaxBudgetUpdate(GPT_4O, {})).toEqual(GPT_4O);
  });

  // Omitting the key would leave the deleted row enforcing, so a cleared editor
  // has to send an explicit empty map.
  it("sends {} when the last row is removed from a budget that was stored", () => {
    expect(modelMaxBudgetUpdate({}, GPT_4O)).toEqual({});
  });

  it.each([
    ["null", null],
    ["undefined", undefined],
    ["empty", {}],
  ])("omits the key entirely when nothing was stored (%s) and nothing was entered", (_label, stored) => {
    expect(modelMaxBudgetUpdate({}, stored)).toBeUndefined();
  });

  // Re-sending an unchanged budget is not merely wasteful: /key/update validates
  // the field whenever it is present and rejects it without an enterprise
  // license, so editing an unrelated field would start failing with a 400.
  describe("omits an unchanged budget so an unrelated edit does not trip the license check", () => {
    it("when the stored value is byte-identical", () => {
      expect(modelMaxBudgetUpdate(GPT_4O, { "gpt-4o": { budget_limit: 5, time_period: "30d" } })).toBeUndefined();
    });

    it("when the proxy stored it under the BudgetConfig aliases", () => {
      expect(modelMaxBudgetUpdate(GPT_4O, { "gpt-4o": { max_budget: 5, budget_duration: "30d" } })).toBeUndefined();
    });

    it("when a CRUD endpoint stored the limit as a string", () => {
      expect(modelMaxBudgetUpdate(GPT_4O, { "gpt-4o": { budget_limit: "5", time_period: "30d" } })).toBeUndefined();
    });

    it("when only the key order differs", () => {
      const edited = {
        "gpt-4o": { budget_limit: 5, time_period: "30d" },
        "claude-opus-4-8": { budget_limit: 1, time_period: "1h" },
      };
      expect(
        modelMaxBudgetUpdate(edited, {
          "claude-opus-4-8": { budget_limit: 1, time_period: "1h" },
          "gpt-4o": { budget_limit: 5, time_period: "30d" },
        }),
      ).toBeUndefined();
    });
  });

  describe("sends the edited budgets whenever anything actually changed", () => {
    it.each([
      ["the limit", { "gpt-4o": { budget_limit: 6, time_period: "30d" } }],
      ["the period", { "gpt-4o": { budget_limit: 5, time_period: "7d" } }],
      ["the model", { "gpt-4o-mini": { budget_limit: 5, time_period: "30d" } }],
      ["an added model", { ...GPT_4O, "claude-opus-4-8": { budget_limit: 1, time_period: "1h" } }],
    ])("%s", (_label, stored) => {
      expect(modelMaxBudgetUpdate(GPT_4O, stored)).toEqual(GPT_4O);
    });

    // 0 is a real cap, so it must not compare equal to "no limit stored".
    it("a zero cap replacing a stored budget with no limit at all", () => {
      const zeroed = { "gpt-4o": { budget_limit: 0, time_period: "30d" } };
      expect(modelMaxBudgetUpdate(zeroed, { "gpt-4o": { time_period: "30d" } })).toEqual(zeroed);
    });
  });
});
