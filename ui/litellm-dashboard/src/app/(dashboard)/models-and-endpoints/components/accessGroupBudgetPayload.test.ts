import { describe, expect, it } from "vitest";

import { accessGroupBudgetFormValues, buildAccessGroupBudgetBody, hasAnyBudgetValue } from "./accessGroupBudgetPayload";

describe("accessGroupBudgetFormValues", () => {
  it("gives every field an empty string when the group has no budget", () => {
    expect(accessGroupBudgetFormValues(null)).toEqual({
      max_budget: "",
      soft_budget: "",
      budget_duration: "",
    });
  });

  it("fills the form from a stored budget", () => {
    expect(
      accessGroupBudgetFormValues({
        budget_id: "budget-1",
        max_budget: 2.5,
        soft_budget: 1,
        budget_duration: "30d",
        budget_reset_at: null,
      }),
    ).toEqual({ max_budget: "2.5", soft_budget: "1", budget_duration: "30d" });
  });

  it("shows a zero max budget rather than treating it as unset", () => {
    expect(accessGroupBudgetFormValues({ budget_id: "budget-1", max_budget: 0 }).max_budget).toBe("0");
  });
});

describe("buildAccessGroupBudgetBody", () => {
  it("sends numbers, not the strings the inputs hold", () => {
    expect(buildAccessGroupBudgetBody({ max_budget: "2.5", soft_budget: "1", budget_duration: "30d" })).toEqual({
      max_budget: 2.5,
      soft_budget: 1,
      budget_duration: "30d",
    });
  });

  it("leaves a blank field out entirely, because the proxy ignores an explicit null", () => {
    const body = buildAccessGroupBudgetBody({ max_budget: "10", soft_budget: "", budget_duration: "" });

    expect(body).toEqual({ max_budget: 10 });
    expect(body).not.toHaveProperty("soft_budget");
    expect(body).not.toHaveProperty("budget_duration");
  });

  it("sends a reset window on its own", () => {
    expect(buildAccessGroupBudgetBody({ budget_duration: "7d" })).toEqual({ budget_duration: "7d" });
  });
});

describe("hasAnyBudgetValue", () => {
  it("rejects a form where every field is blank, which the proxy answers with a 400", () => {
    expect(hasAnyBudgetValue({ max_budget: "", soft_budget: "", budget_duration: "" })).toBe(false);
    expect(hasAnyBudgetValue({})).toBe(false);
  });

  it("accepts a form with any one field filled", () => {
    expect(hasAnyBudgetValue({ soft_budget: "1" })).toBe(true);
  });
});
