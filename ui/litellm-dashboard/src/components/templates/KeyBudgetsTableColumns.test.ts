import { describe, expect, it } from "vitest";
import type { KeyBudgetEntry } from "@/app/(dashboard)/hooks/keys/useKeyBudgets";
import { budgetThresholdRule, isBlockingRow, severityRank } from "./KeyBudgetsTableColumns";

const UNCONFIGURED_BUDGET = {
  scope: "key",
  entity_type: "key",
  entity_id: null,
  entity_label: null,
  enforcement: "hard",
  max_budget: null,
  spend: 0,
  remaining: null,
  comparison: ">=",
  budget_duration: null,
  budget_reset_at: null,
  window_start: null,
  source: "key.max_budget",
  status: "unlimited",
  note: null,
} as KeyBudgetEntry;

const TEAM_MEMBER_AT_LIMIT: KeyBudgetEntry = {
  ...UNCONFIGURED_BUDGET,
  scope: "team_member",
  comparison: ">=",
  max_budget: 50,
  spend: 50,
  remaining: 0,
  status: "exceeded",
};

const PROJECT_UNDER_LIMIT: KeyBudgetEntry = {
  ...UNCONFIGURED_BUDGET,
  scope: "project",
  comparison: ">",
  max_budget: 300,
  spend: 120,
  remaining: 180,
  status: "ok",
};

// team is ">=" while budget reservation is on and ">" once an operator disables it, so the same
// scope must render either operator off the response rather than a value baked in per scope.
const RESERVED_TEAM_AT_300: KeyBudgetEntry = {
  ...UNCONFIGURED_BUDGET,
  scope: "team",
  comparison: ">=",
  max_budget: 300,
  spend: 300,
  remaining: 0,
  status: "exceeded",
};

const UNRESERVED_TEAM_AT_300: KeyBudgetEntry = { ...RESERVED_TEAM_AT_300, comparison: ">", status: "ok" };

const SOFT_OVER: KeyBudgetEntry = {
  ...UNCONFIGURED_BUDGET,
  enforcement: "soft",
  comparison: ">=",
  max_budget: 500,
  spend: 900,
  remaining: -400,
  status: "exceeded",
};

const SUB_DOLLAR_LIMIT: KeyBudgetEntry = { ...UNCONFIGURED_BUDGET, comparison: ">", max_budget: 0.1 };

const BLOCKING: KeyBudgetEntry = { ...UNCONFIGURED_BUDGET, status: "exceeded", enforcement: "hard" };
const ALERT_ONLY: KeyBudgetEntry = { ...UNCONFIGURED_BUDGET, status: "exceeded", enforcement: "soft" };
const HEALTHY: KeyBudgetEntry = { ...UNCONFIGURED_BUDGET, status: "ok", enforcement: "hard" };

describe("budgetThresholdRule", () => {
  it("marks an inclusive scope as blocking at the limit, so spend equal to it is already denied", () => {
    expect(budgetThresholdRule(TEAM_MEMBER_AT_LIMIT)).toBe("Blocks at ≥ $50.00");
  });

  it("marks an exclusive scope as blocking only above the limit", () => {
    expect(budgetThresholdRule(PROJECT_UNDER_LIMIT)).toBe("Blocks at > $300.00");
  });

  it("reads the operator off each response, so one scope can render either threshold", () => {
    expect(budgetThresholdRule(RESERVED_TEAM_AT_300)).toBe("Blocks at ≥ $300.00");
    expect(budgetThresholdRule(UNRESERVED_TEAM_AT_300)).toBe("Blocks at > $300.00");
  });

  it("never promises a soft budget will block", () => {
    expect(budgetThresholdRule(SOFT_OVER)).toBe("Alerts at ≥ $500.00");
  });

  it("states no threshold for a scope with nothing configured", () => {
    expect(budgetThresholdRule(UNCONFIGURED_BUDGET)).toBeNull();
  });

  it("keeps sub-dollar limits legible rather than rounding them to zero", () => {
    expect(budgetThresholdRule(SUB_DOLLAR_LIMIT)).toBe("Blocks at > $0.10");
  });
});

describe("isBlockingRow", () => {
  it("counts only a hard budget that is over as blocking", () => {
    expect(isBlockingRow(BLOCKING)).toBe(true);
    expect(isBlockingRow(ALERT_ONLY)).toBe(false);
    expect(isBlockingRow(HEALTHY)).toBe(false);
    expect(isBlockingRow(UNCONFIGURED_BUDGET)).toBe(false);
  });
});

describe("severityRank", () => {
  it("ranks a blocking budget above an alert-only one, and both above healthy and unlimited", () => {
    expect(severityRank(BLOCKING)).toBeLessThan(severityRank(ALERT_ONLY));
    expect(severityRank(ALERT_ONLY)).toBeLessThan(severityRank(HEALTHY));
    expect(severityRank(HEALTHY)).toBeLessThan(severityRank(UNCONFIGURED_BUDGET));
  });
});
