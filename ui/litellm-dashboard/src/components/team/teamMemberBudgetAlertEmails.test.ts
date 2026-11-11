import { describe, expect, it } from "vitest";
import {
  isValidThreshold,
  teamMemberBudgetAlertEmailsFromRows,
  teamMemberBudgetAlertRowsFromMetadata,
  teamMemberBudgetAlertSummary,
} from "./teamMemberBudgetAlertEmails";

describe("teamMemberBudgetAlertRowsFromMetadata", () => {
  it("turns the stored threshold map into rows sorted by threshold", () => {
    const metadata = {
      team_member_max_budget_alert_emails: { "100": ["finance@example.com", "cto@example.com"], "50": [] },
    };
    expect(teamMemberBudgetAlertRowsFromMetadata(metadata)).toEqual([
      { threshold: 50, emails: "" },
      { threshold: 100, emails: "finance@example.com, cto@example.com" },
    ]);
  });

  it("drops non-numeric thresholds and non-list recipients instead of crashing", () => {
    const metadata = {
      team_member_max_budget_alert_emails: { fifty: [], "75": "finance@example.com", "90": [1], "100": ["a@b.c"] },
    };
    expect(teamMemberBudgetAlertRowsFromMetadata(metadata)).toEqual([{ threshold: 100, emails: "a@b.c" }]);
  });

  it("drops API-stored thresholds outside 1 to 100 so they never block the form", () => {
    const metadata = {
      team_member_max_budget_alert_emails: { "0": ["a@b.c"], "50": [], "101": ["a@b.c"] },
    };
    expect(teamMemberBudgetAlertRowsFromMetadata(metadata)).toEqual([{ threshold: 50, emails: "" }]);
  });

  it.each([undefined, null, "50", { team_member_max_budget_alert_emails: "50" }, { soft_budget_alerting_emails: [] }])(
    "returns no rows for unrelated or malformed metadata %j",
    (metadata) => {
      expect(teamMemberBudgetAlertRowsFromMetadata(metadata)).toEqual([]);
    },
  );
});

describe("teamMemberBudgetAlertEmailsFromRows", () => {
  it("builds the threshold map, splitting, trimming and deduplicating recipients", () => {
    expect(
      teamMemberBudgetAlertEmailsFromRows([
        { threshold: 50, emails: "" },
        { threshold: 100, emails: " finance@example.com,cto@example.com , finance@example.com, " },
      ]),
    ).toEqual({ "50": [], "100": ["finance@example.com", "cto@example.com"] });
  });

  it("skips rows without a valid threshold", () => {
    expect(
      teamMemberBudgetAlertEmailsFromRows([
        { threshold: null, emails: "finance@example.com" },
        { threshold: 0, emails: "" },
        { threshold: 101, emails: "" },
        { threshold: 12.5, emails: "" },
        { threshold: 80, emails: "" },
      ]),
    ).toEqual({ "80": [] });
  });

  it("round-trips the stored config", () => {
    const stored = { team_member_max_budget_alert_emails: { "50": [], "100": ["finance@example.com"] } };
    expect(teamMemberBudgetAlertEmailsFromRows(teamMemberBudgetAlertRowsFromMetadata(stored))).toEqual(
      stored.team_member_max_budget_alert_emails,
    );
  });
});

describe("isValidThreshold", () => {
  it.each([
    [1, true],
    [50, true],
    [100, true],
    [0, false],
    [101, false],
    [33.3, false],
    [null, false],
  ])("treats %s as valid=%s", (threshold, valid) => {
    expect(isValidThreshold(threshold)).toBe(valid);
  });
});

describe("teamMemberBudgetAlertSummary", () => {
  it("states that the member is always notified and lists extra recipients", () => {
    expect(
      teamMemberBudgetAlertSummary({
        team_member_max_budget_alert_emails: { "100": ["finance@example.com"], "50": [] },
      }),
    ).toEqual(["50%: member", "100%: member, finance@example.com"]);
  });
});
