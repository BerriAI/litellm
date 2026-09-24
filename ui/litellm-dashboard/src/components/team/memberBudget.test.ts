import { describe, expect, it } from "vitest";
import type { TeamMembership } from "./TeamInfo";
import { displayedMemberBudget } from "./memberBudget";

const teamDefault = {
  max_budget: 1000,
  budget_duration: "1mo",
  budget_reset_at: "2100-02-01T00:00:00Z",
  rpm_limit: 2,
  tpm_limit: 100,
};
const membership: TeamMembership = {
  user_id: "member",
  team_id: "team",
  budget_id: "temporary",
  budget_source: "custom",
  spend: 1100,
  total_spend: 1100,
  litellm_budget_table: {
    budget_id: "temporary",
    max_budget: null,
    soft_budget: null,
    max_parallel_requests: null,
    rpm_limit: null,
    tpm_limit: null,
    model_max_budget: null,
    budget_duration: null,
    budget_reset_at: null,
    temp_budget_increase: 200,
    temp_budget_expiry: "2100-02-01T00:00:00Z",
  },
};

describe("displayedMemberBudget", () => {
  it.each(["2020-02-01T00:00:00Z", "2100-02-01T00:00:00Z"])(
    "shows inherited rates, cap and reset for a grant expiring at %s without changing edit values",
    (expiry) => {
      const member = {
        ...membership,
        litellm_budget_table: { ...membership.litellm_budget_table, temp_budget_expiry: expiry },
      };
      expect(displayedMemberBudget(member, teamDefault)).toEqual(teamDefault);
      expect(member.litellm_budget_table.max_budget).toBeNull();
      expect(member.litellm_budget_table.budget_duration).toBeNull();
    },
  );

  it.each([
    { max_budget: 0 },
    { rpm_limit: 1 },
    { tpm_limit: 50 },
    { tpd_limit: 10 },
    { soft_budget: 50 },
    { max_parallel_requests: 1 },
    { budget_duration: "1d" },
    { model_max_budget: { model: 1 } },
    { allowed_models: ["model"] },
    { temp_budget_increase: null },
  ])("keeps an explicit member budget independent: %j", (override) => {
    const member = {
      ...membership,
      litellm_budget_table: { ...membership.litellm_budget_table, ...override },
    };
    expect(displayedMemberBudget(member, teamDefault)).toBe(member.litellm_budget_table);
  });
});
