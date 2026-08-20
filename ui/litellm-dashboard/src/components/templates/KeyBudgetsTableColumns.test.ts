import { describe, expect, it } from "vitest";
import type { KeyBudgetEntry, KeyBudgetNote } from "@/app/(dashboard)/hooks/keys/useKeyBudgets";
import { budgetThresholdRule, cannotTrip, isBlockingRow, isThrottled, rowRank } from "./KeyBudgetsTableColumns";

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
  spend_state: "live",
  notes: [],
} as KeyBudgetEntry;

// `code` and `severity` are the contract; `text` is explicitly free to be reworded, so these are
// synthetic. A test that pins the server's prose asserts the one field promised to change.
const noteText = (code: string): string => `synthetic ${code} caveat`;

const PROJECT_DEAD_NOTE = {
  code: "project_spend_not_tracked",
  severity: "warning",
  text: noteText("project_spend_not_tracked"),
} as const;

const ALERT_ONLY_NOTE = {
  code: "alert_only",
  severity: "info",
  text: noteText("alert_only"),
} as const;

const ROLLING_NOTE = {
  code: "rolling_window",
  severity: "info",
  text: noteText("rolling_window"),
} as const;

// `enforcement` carries the fact now, so the note only explains the mechanism and drops to info.
const THROTTLE_NOTE = {
  code: "throttled_instead_of_blocked",
  severity: "info",
  text: noteText("throttled_instead_of_blocked"),
} as const;

const END_USER_ROUTE_NOTE = {
  code: "end_user_route_only",
  severity: "warning",
  text: noteText("end_user_route_only"),
} as const;

// Permanently inert for this key: the personal budget is not applied to team keys at all.
const USER_ON_TEAM_KEY_NOTE = {
  code: "user_budget_not_applied_to_team_key",
  severity: "info",
  text: noteText("user_budget_not_applied_to_team_key"),
} as const;

// Transient, not dead: the counter is merely cold and the cap blocks once one request warms it.
const COLD_MODEL_NOTE = {
  code: "model_budget_fails_open",
  severity: "info",
  text: noteText("model_budget_fails_open"),
} as const;

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
const ALERT_ONLY: KeyBudgetEntry = {
  ...UNCONFIGURED_BUDGET,
  status: "exceeded",
  enforcement: "soft",
  notes: [ALERT_ONLY_NOTE],
};
const HEALTHY: KeyBudgetEntry = { ...UNCONFIGURED_BUDGET, status: "ok", enforcement: "hard" };
const INERT: KeyBudgetEntry = {
  ...UNCONFIGURED_BUDGET,
  scope: "project",
  status: "ok",
  max_budget: 300,
  spend: 0,
  remaining: 300,
  notes: [PROJECT_DEAD_NOTE],
};
const WARNED: KeyBudgetEntry = { ...HEALTHY, notes: [ROLLING_NOTE] };

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

describe("cannotTrip", () => {
  it("treats an info note as proof the row is dead", () => {
    expect(cannotTrip(INERT)).toBe(true);
  });

  it("does not call a soft budget dead, since alert_only only restates the enforcement column", () => {
    expect(cannotTrip(ALERT_ONLY)).toBe(false);
  });

  it("leaves a warning note trippable, because it qualifies the number rather than killing it", () => {
    expect(cannotTrip(WARNED)).toBe(false);
  });

  // Severity answers whether the row already carries the fact in a field, which is orthogonal to
  // whether the row is dead. All four corners occur, so severity can never stand in for deadness.
  it.each([
    ["info", "dead", USER_ON_TEAM_KEY_NOTE, true],
    ["info", "live", ROLLING_NOTE, false],
    ["warning", "dead", PROJECT_DEAD_NOTE, true],
    ["warning", "live", END_USER_ROUTE_NOTE, false],
  ])("treats a %s note that is %s by code, not by severity", (severity, _kind, note, dead) => {
    expect(note.severity).toBe(severity);
    expect(cannotTrip({ ...HEALTHY, notes: [note] })).toBe(dead);
  });

  it("assumes a code this build predates is live, so an unknown caveat never hides a real blocker", () => {
    const future = { code: "some_code_added_later", severity: "info", text: "…" } as unknown as KeyBudgetNote;
    const louder = { ...future, severity: "warning" } as KeyBudgetNote;
    expect(cannotTrip({ ...HEALTHY, notes: [future] })).toBe(false);
    expect(cannotTrip({ ...HEALTHY, notes: [louder] })).toBe(false);
  });

  it("branches on code and severity rather than on wording, so text is free to be reworded", () => {
    const reworded: KeyBudgetEntry = {
      ...INERT,
      notes: [{ ...PROJECT_DEAD_NOTE, text: "totally different prose that never mentions tripping" }],
    };
    expect(cannotTrip(reworded)).toBe(true);
  });
});

describe("cold per-model counters", () => {
  it("stays live, because the cap blocks as soon as one request warms the counter", () => {
    expect(COLD_MODEL_NOTE.severity).toBe("info");
    expect(cannotTrip({ ...HEALTHY, scope: "key_model", notes: [COLD_MODEL_NOTE] })).toBe(false);
  });

  it("is not lumped in with a budget that is permanently inert", () => {
    expect(cannotTrip({ ...HEALTHY, notes: [USER_ON_TEAM_KEY_NOTE] })).toBe(true);
    expect(rowRank({ ...HEALTHY, scope: "key_model", notes: [COLD_MODEL_NOTE] })).toBeLessThan(
      rowRank({ ...HEALTHY, notes: [USER_ON_TEAM_KEY_NOTE] }),
    );
  });
});

describe("isThrottled", () => {
  it("never blames a throttled budget for a denial, because going over slows rather than rejects", () => {
    const throttledOver: KeyBudgetEntry = {
      ...UNCONFIGURED_BUDGET,
      enforcement: "throttled",
      max_budget: 100,
      spend: 140,
      remaining: -40,
      status: "exceeded",
      notes: [THROTTLE_NOTE],
    };
    expect(isThrottled(throttledOver)).toBe(true);
    expect(isBlockingRow(throttledOver)).toBe(false);
    expect(budgetThresholdRule(throttledOver)).toBe("Throttles at ≥ $100.00");
  });

  it("reads the mode off enforcement rather than the note, which only explains the mechanism", () => {
    const noNote: KeyBudgetEntry = { ...UNCONFIGURED_BUDGET, enforcement: "throttled", max_budget: 100, notes: [] };
    expect(isThrottled(noNote)).toBe(true);
    const noteOnly: KeyBudgetEntry = { ...UNCONFIGURED_BUDGET, enforcement: "hard", notes: [THROTTLE_NOTE] };
    expect(isThrottled(noteOnly)).toBe(false);
  });

  it("still says a plain hard budget blocks", () => {
    expect(isThrottled(TEAM_MEMBER_AT_LIMIT)).toBe(false);
    expect(budgetThresholdRule(TEAM_MEMBER_AT_LIMIT)).toBe("Blocks at ≥ $50.00");
  });
});

describe("isBlockingRow", () => {
  it("counts only a hard budget that is over as blocking", () => {
    expect(isBlockingRow(BLOCKING)).toBe(true);
    expect(isBlockingRow(ALERT_ONLY)).toBe(false);
    expect(isBlockingRow(HEALTHY)).toBe(false);
    expect(isBlockingRow(UNCONFIGURED_BUDGET)).toBe(false);
  });

  it("never blames a budget that structurally cannot trip, however over it looks", () => {
    const overButDead: KeyBudgetEntry = { ...INERT, spend: 900, remaining: -600, status: "exceeded" };
    expect(overButDead.status).toBe("exceeded");
    expect(isBlockingRow(overButDead)).toBe(false);
  });
});

describe("rowRank", () => {
  it("ranks a blocking budget above an alert-only one, and both above healthy and unlimited", () => {
    expect(rowRank(BLOCKING)).toBeLessThan(rowRank(ALERT_ONLY));
    expect(rowRank(ALERT_ONLY)).toBeLessThan(rowRank(HEALTHY));
    expect(rowRank(HEALTHY)).toBeLessThan(rowRank(UNCONFIGURED_BUDGET));
  });

  it("sinks a row that cannot trip below every row that can, since it never stopped anything", () => {
    expect(rowRank(INERT)).toBeGreaterThan(rowRank(BLOCKING));
    expect(rowRank(INERT)).toBeGreaterThan(rowRank(ALERT_ONLY));
    expect(rowRank(INERT)).toBeGreaterThan(rowRank(HEALTHY));
    expect(rowRank(INERT)).toBeGreaterThan(rowRank(UNCONFIGURED_BUDGET));
  });
});
