import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import type { KeyBudgetEntry } from "@/app/(dashboard)/hooks/keys/useKeyBudgets";

import { KeyBudgetsBulletChart, plottable, utilization } from "./KeyBudgetsBulletChart";

// The chart answers "what stopped my request" before the table lists the evidence, so every test
// here is about a wrong answer being worse than no answer: a scope ruled out that should not be.

const BASE = {
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
  spend_state: "live",
  notes: [],
} as KeyBudgetEntry;

const OK: KeyBudgetEntry = { ...BASE, status: "ok" };

const TEAM_HALF: KeyBudgetEntry = { ...OK, scope: "team", max_budget: 100, spend: 50, remaining: 50 };
const ORG_NEARLY: KeyBudgetEntry = { ...OK, scope: "organization", max_budget: 100, spend: 88, remaining: 12 };
const USER_TENTH: KeyBudgetEntry = { ...OK, scope: "user", max_budget: 100, spend: 10, remaining: 90 };
const MEMBER_OVER: KeyBudgetEntry = {
  ...OK,
  scope: "team_member",
  max_budget: 100,
  spend: 140,
  remaining: -40,
  status: "exceeded",
};
const UNLIMITED_KEY: KeyBudgetEntry = { ...OK, spend: 900, status: "unlimited" };
const ZERO_CAP: KeyBudgetEntry = { ...OK, max_budget: 0, spend: 5, remaining: -5 };
const TINY_SHARE: KeyBudgetEntry = { ...OK, scope: "tag", max_budget: 1000, spend: 0.4, remaining: 999.6 };
const UNREADABLE_TEAM: KeyBudgetEntry = {
  ...OK,
  scope: "team",
  spend: null,
  spend_state: "unavailable",
  status: "unknown",
  notes: [{ code: "entity_unavailable", severity: "warning", text: "could not read" }],
};
const RESTRICTED_PROXY: KeyBudgetEntry = {
  ...OK,
  scope: "proxy",
  spend: null,
  spend_state: "restricted",
  status: "unknown",
  notes: [{ code: "proxy_spend_restricted", severity: "warning", text: "admins only" }],
};
const DEAD_PROJECT: KeyBudgetEntry = {
  ...OK,
  scope: "project",
  max_budget: 100,
  spend: 99,
  remaining: 1,
  notes: [{ code: "project_spend_not_tracked", severity: "warning", text: "never increments" }],
};

const verdict = () => screen.getByTestId("key-budgets-verdict");

describe("utilization", () => {
  it("is the fraction of the limit spent", () => {
    expect(utilization(ORG_NEARLY)).toBeCloseTo(0.88, 6);
  });

  it("refuses to invent a fraction for a row with no limit or no reading", () => {
    expect(utilization(UNLIMITED_KEY)).toBeNull();
    expect(utilization(UNREADABLE_TEAM)).toBeNull();
    // A zero cap is "unset" to every enforcing check, so dividing by it would plot a phantom bar.
    expect(utilization(ZERO_CAP)).toBeNull();
  });
});

describe("plottable", () => {
  it("orders by how close each budget is to its limit, worst first", () => {
    const ordered = plottable([USER_TENTH, ORG_NEARLY, TEAM_HALF]).map((entry) => entry.scope);
    expect(ordered).toStrictEqual(["organization", "team", "user"]);
  });

  it("leaves out a budget that structurally cannot trip, however close to its cap it looks", () => {
    expect(utilization(DEAD_PROJECT)).toBeCloseTo(0.99, 6);
    expect(plottable([DEAD_PROJECT, TEAM_HALF]).map((entry) => entry.scope)).toStrictEqual(["team"]);
  });

  it("leaves out rows with nothing to draw rather than drawing them at zero", () => {
    expect(plottable([UNLIMITED_KEY, UNREADABLE_TEAM])).toHaveLength(0);
  });
});

describe("KeyBudgetsBulletChart", () => {
  it("names the budget that is blocking, not merely that something is", () => {
    render(<KeyBudgetsBulletChart budgets={[TEAM_HALF, MEMBER_OVER, UNLIMITED_KEY]} />);

    expect(verdict()).toHaveTextContent("Blocked by Team member");
    expect(verdict()).not.toHaveTextContent("Nothing is blocking");
    expect(screen.getByTestId("key-budget-bullet-blocking")).toBeInTheDocument();
  });

  it("names the budget closest to its limit when nothing is blocking", () => {
    render(<KeyBudgetsBulletChart budgets={[USER_TENTH, ORG_NEARLY, TEAM_HALF]} />);

    expect(verdict()).toHaveTextContent("Nothing is blocking this key.");
    expect(verdict()).toHaveTextContent("Closest to its limit: Organization, 88% used.");
    expect(screen.queryByTestId("key-budget-bullet-blocking")).not.toBeInTheDocument();
  });

  it("names a scope nobody could evaluate, since a verdict that ignores it is a verdict ruling it out", () => {
    render(<KeyBudgetsBulletChart budgets={[TEAM_HALF, UNREADABLE_TEAM]} />);

    expect(verdict()).toHaveTextContent("1 scope could not be evaluated");
    expect(verdict()).toHaveTextContent("Team");
  });

  it("names a scope whose numbers this caller may not read, which is no more ruled out than a failed read", () => {
    render(<KeyBudgetsBulletChart budgets={[TEAM_HALF, RESTRICTED_PROXY]} />);

    expect(verdict()).toHaveTextContent("1 scope could not be evaluated");
    expect(verdict()).toHaveTextContent("Proxy");
  });

  it("draws each bar in proportion to its budget, and never past the end of its track", () => {
    render(<KeyBudgetsBulletChart budgets={[ORG_NEARLY, MEMBER_OVER]} />);

    const [over, nearly] = [screen.getByTestId("key-budget-bullet-blocking"), screen.getByTestId("key-budget-bullet")];
    expect(nearly).toHaveStyle({ width: "88%" });
    // 140% of the cap would otherwise render as a bar overflowing its own track.
    expect(over).toHaveStyle({ width: "100%" });
  });

  it("keeps a sub-percent balance visible instead of rounding it away to 0%", () => {
    render(<KeyBudgetsBulletChart budgets={[TINY_SHARE]} />);

    expect(screen.getByTestId("key-budgets-chart")).toHaveTextContent("<1%");
  });

  it("plots nothing at all rather than an empty grid when no budget carries a limit", () => {
    render(<KeyBudgetsBulletChart budgets={[UNLIMITED_KEY]} />);

    const chart = screen.getByTestId("key-budgets-chart");
    expect(within(chart).queryByTestId("key-budget-bullet")).not.toBeInTheDocument();
    expect(verdict()).toHaveTextContent("Nothing is blocking this key.");
    expect(verdict()).not.toHaveTextContent("Closest to its limit");
  });
});
