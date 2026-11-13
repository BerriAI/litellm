import { describe, expect, it } from "vitest";
import { budgetProfilePayload, type ProfileFormValues } from "./budgetProfilePayload";

describe("budgetProfilePayload", () => {
  it("creates independent caps, including zero, without shared-budget fields", () => {
    const values: ProfileFormValues = {
      budget_id: "starter",
      is_profile: true,
      monthly: 0,
      annual: 100,
      cap_policy: "pause_allow_increase",
      max_budget: 77,
      budget_duration: "7d",
    };
    const expected = {
      budget_id: "starter",
      is_profile: true,
      budget_limits: [
        { budget_duration: "1mo", max_budget: 0 },
        { budget_duration: "12mo", max_budget: 100 },
      ],
      cap_policy: "pause_allow_increase",
    };
    expect(budgetProfilePayload(values)).toEqual(expected);
  });
  it("updates only guidance with the loaded revision", () => {
    const values: ProfileFormValues = {
      budget_id: "starter",
      is_profile: true,
      revision: "loaded",
      monthly: 10,
      annual: 100,
      cap_policy: "pause_until_reset",
    };
    expect(budgetProfilePayload(values, { cap_policy: true })).toEqual({
      budget_id: "starter",
      revision: "loaded",
      cap_policy: "pause_until_reset",
    });
  });
  it("clears one profile cap without deriving or scaling the other", () => {
    const values = { budget_id: "starter", is_profile: true, revision: "loaded", monthly: null, annual: 7 };
    expect(budgetProfilePayload(values, { monthly: true })).toEqual({
      budget_id: "starter",
      revision: "loaded",
      budget_limits: [{ budget_duration: "12mo", max_budget: 7 }],
    });
  });
  it("retains shared-budget precision and dirty-only edits", () => {
    const values = { budget_id: "shared", max_budget: 12.345, budget_duration: "7d", rpm_limit: 8 };
    expect(budgetProfilePayload(values, { max_budget: true })).toEqual({ budget_id: "shared", max_budget: 12.35 });
  });
});
