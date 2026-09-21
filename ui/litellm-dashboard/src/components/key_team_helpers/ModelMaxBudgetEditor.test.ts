import { describe, expect, it } from "vitest";
import { entriesToModelMaxBudget, modelMaxBudgetToEntries, type ModelMaxBudget } from "./ModelMaxBudgetEditor";

describe("modelMaxBudgetToEntries", () => {
  it("hydrates an existing budget without losing its period", () => {
    const budget: ModelMaxBudget = {
      "claude-opus-4-8": { budget_limit: 200, time_period: "1mo" },
      "gpt-4o": { budget_limit: 0.5, time_period: "7d" },
    };
    expect(modelMaxBudgetToEntries(budget)).toEqual([
      { id: "existing-0", model: "claude-opus-4-8", budgetLimit: 200, timePeriod: "1mo", extra: {} },
      { id: "existing-1", model: "gpt-4o", budgetLimit: 0.5, timePeriod: "7d", extra: {} },
    ]);
  });

  it.each([
    ["null", null],
    ["undefined", undefined],
    ["empty", {} as ModelMaxBudget],
  ])("treats %s as no rows", (_label, budget) => {
    expect(modelMaxBudgetToEntries(budget)).toEqual([]);
  });
});

// model_max_budget is a plain dict on keys and users, stored and returned exactly
// as the client sent it, and BudgetConfig documents the max_budget/budget_duration
// spelling. Reading only one spelling mounts the row blank, and emitting then drops
// it, so opening a form and saving it untouched would wipe the stored budget.
describe("modelMaxBudgetToEntries reads either BudgetConfig spelling", () => {
  it.each([
    ["budget_limit/time_period", { budget_limit: 200, time_period: "1mo" }],
    ["max_budget/budget_duration", { max_budget: 200, budget_duration: "1mo" }],
  ])("hydrates a row stored as %s", (_label, config) => {
    expect(modelMaxBudgetToEntries({ "claude-opus-4-8": config })).toEqual([
      { id: "existing-0", model: "claude-opus-4-8", budgetLimit: 200, timePeriod: "1mo", extra: {} },
    ]);
  });

  // /key/update and /budget/new both accept the limit as a string.
  it.each([
    ["a plain number", 0.5, 0.5],
    ["a numeric string", "0.5", 0.5],
    ["a trailing-zero string", "0.50", 0.5],
    ["exponent notation", "5e-1", 0.5],
    ["zero, which is a real cap", 0, 0],
  ])("reads %s as the cap", (_label, stored, expected) => {
    expect(modelMaxBudgetToEntries({ "gpt-4o": { budget_limit: stored, time_period: "1h" } })[0].budgetLimit).toBe(
      expected,
    );
  });

  it("leaves the cap empty rather than NaN when the stored value is not a number", () => {
    expect(
      modelMaxBudgetToEntries({ "gpt-4o": { budget_limit: "not a number", time_period: "1h" } })[0].budgetLimit,
    ).toBeNull();
  });

  it("falls back to the default period rather than an empty one", () => {
    expect(modelMaxBudgetToEntries({ "gpt-4o": { budget_limit: 1, time_period: "" } })[0].timePeriod).toBe("30d");
  });
});

// BudgetConfig carries tpm_limit and rpm_limit too. This editor models neither, so
// without carrying them through, editing a dollar cap silently drops a configured
// rate limit.
describe("fields the editor does not model", () => {
  const WITH_RATE_LIMITS: ModelMaxBudget = {
    "gpt-4o": { budget_limit: 5, time_period: "1h", tpm_limit: 1000, rpm_limit: 60 },
  };

  it("keeps them on the entry when hydrating", () => {
    expect(modelMaxBudgetToEntries(WITH_RATE_LIMITS)[0].extra).toEqual({ tpm_limit: 1000, rpm_limit: 60 });
  });

  it("puts them back when the cap is edited", () => {
    const edited = modelMaxBudgetToEntries(WITH_RATE_LIMITS).map((entry) => ({ ...entry, budgetLimit: 9 }));

    expect(entriesToModelMaxBudget(edited)).toEqual({
      "gpt-4o": { budget_limit: 9, time_period: "1h", tpm_limit: 1000, rpm_limit: 60 },
    });
  });

  // Emitting both spellings would leave the proxy with a contradictory config.
  it("does not re-emit the alias spelling alongside the canonical one", () => {
    const stored: ModelMaxBudget = { "gpt-4o": { max_budget: 5, budget_duration: "1h", tpm_limit: 1000 } };

    expect(entriesToModelMaxBudget(modelMaxBudgetToEntries(stored))).toEqual({
      "gpt-4o": { budget_limit: 5, time_period: "1h", tpm_limit: 1000 },
    });
  });
});

describe("entriesToModelMaxBudget", () => {
  // Two models are two independent budgets, so neither row order nor a
  // half-filled row may change which budgets are submitted.
  it.each([
    ["configured first", ["gpt-4o", null] as const],
    ["configured second", [null, "gpt-4o"] as const],
  ])("drops a row with no model, %s", (_label, models) => {
    const entries = models.map((model, index) => ({
      id: String(index),
      model,
      budgetLimit: 1.25,
      timePeriod: "30d",
      extra: {},
    }));
    expect(entriesToModelMaxBudget(entries)).toEqual({
      "gpt-4o": { budget_limit: 1.25, time_period: "30d" },
    });
  });

  it("drops a row whose budget was never typed", () => {
    expect(
      entriesToModelMaxBudget([
        { id: "1", model: "gpt-4o", budgetLimit: null, timePeriod: "30d", extra: {} },
        { id: "2", model: "claude-opus-4-8", budgetLimit: 3, timePeriod: "1h", extra: {} },
      ]),
    ).toEqual({ "claude-opus-4-8": { budget_limit: 3, time_period: "1h" } });
  });

  it("keeps a zero budget, which is a real cap and not an empty field", () => {
    expect(
      entriesToModelMaxBudget([{ id: "1", model: "gpt-4o", budgetLimit: 0, timePeriod: "30d", extra: {} }]),
    ).toEqual({
      "gpt-4o": { budget_limit: 0, time_period: "30d" },
    });
  });

  it("round-trips an existing budget unchanged", () => {
    const budget: ModelMaxBudget = {
      "claude-opus-4-8": { budget_limit: 200, time_period: "1mo" },
      "gpt-4o": { budget_limit: 0.5, time_period: "7d" },
    };
    expect(entriesToModelMaxBudget(modelMaxBudgetToEntries(budget))).toEqual(budget);
  });

  it("submits nothing once the last row is removed", () => {
    expect(entriesToModelMaxBudget([])).toEqual({});
  });
});
