import { describe, expect, it } from "vitest";
import {
  budgetAlertEmailsFromRows,
  budgetAlertEmailsMetadata,
  budgetAlertEmailsRowsCanShow,
  budgetAlertRowsFromMetadata,
  isValidThreshold,
  KEY_MAX_BUDGET_ALERT_EMAILS_KEY,
  TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY,
  teamMemberBudgetAlertSummary,
} from "./budgetAlertEmails";

describe("budgetAlertRowsFromMetadata", () => {
  it("turns the stored threshold map into rows sorted by threshold", () => {
    const metadata = {
      team_member_max_budget_alert_emails: { "100": ["finance@example.com", "cto@example.com"], "50": [] },
    };
    expect(budgetAlertRowsFromMetadata(metadata, TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY)).toEqual([
      { threshold: 50, emails: "" },
      { threshold: 100, emails: "finance@example.com, cto@example.com" },
    ]);
  });

  it("drops non-numeric thresholds and non-email recipients instead of crashing", () => {
    const metadata = {
      team_member_max_budget_alert_emails: { fifty: [], "90": { a: 1 }, "100": ["a@b.c"] },
    };
    expect(budgetAlertRowsFromMetadata(metadata, TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY)).toEqual([
      { threshold: 100, emails: "a@b.c" },
    ]);
  });

  it("loads a null recipient list as an owner-only row and keeps the strings of a mixed list", () => {
    const metadata = { max_budget_alert_emails: { "75": null, "80": ["b@example.com", null, 3] } };
    expect(budgetAlertRowsFromMetadata(metadata, KEY_MAX_BUDGET_ALERT_EMAILS_KEY)).toEqual([
      { threshold: 75, emails: "" },
      { threshold: 80, emails: "b@example.com" },
    ]);
  });

  it("lists API-stored string recipients in the team member summary", () => {
    expect(teamMemberBudgetAlertSummary({ team_member_max_budget_alert_emails: { "80": "a@x.com, b@x.com" } })).toEqual(
      ["80%: member, a@x.com, b@x.com"],
    );
  });

  it("leaves a zero-padded threshold to the Metadata box so it never duplicates a row", () => {
    const metadata = { max_budget_alert_emails: { "080": ["padded@example.com"], "80": ["plain@example.com"] } };
    expect(budgetAlertRowsFromMetadata(metadata, KEY_MAX_BUDGET_ALERT_EMAILS_KEY)).toEqual([
      { threshold: 80, emails: "plain@example.com" },
    ]);
  });

  it("still loads a zero-padded team member threshold as a row, since the team form has no Metadata box fallback", () => {
    const metadata = { team_member_max_budget_alert_emails: { "080": ["finance@example.com"] } };
    expect(budgetAlertRowsFromMetadata(metadata, TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY)).toEqual([
      { threshold: 80, emails: "finance@example.com" },
    ]);
  });

  it("keeps API-stored comma-separated string recipients, which the backend also accepts", () => {
    const metadata = {
      max_budget_alert_emails: { "80": "finance@example.com, cto@example.com", "90": ["ops@example.com"] },
    };
    const rows = budgetAlertRowsFromMetadata(metadata, KEY_MAX_BUDGET_ALERT_EMAILS_KEY);
    expect(rows).toEqual([
      { threshold: 80, emails: "finance@example.com, cto@example.com" },
      { threshold: 90, emails: "ops@example.com" },
    ]);
    expect(budgetAlertEmailsFromRows(rows)).toEqual({
      "80": ["finance@example.com", "cto@example.com"],
      "90": ["ops@example.com"],
    });
  });

  it("drops API-stored thresholds outside 1 to 100 so they never block the form", () => {
    const metadata = {
      team_member_max_budget_alert_emails: { "0": ["a@b.c"], "50": [], "101": ["a@b.c"] },
    };
    expect(budgetAlertRowsFromMetadata(metadata, TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY)).toEqual([
      { threshold: 50, emails: "" },
    ]);
  });

  it.each([undefined, null, "50", { team_member_max_budget_alert_emails: "50" }, { soft_budget_alerting_emails: [] }])(
    "returns no rows for unrelated or malformed metadata %j",
    (metadata) => {
      expect(budgetAlertRowsFromMetadata(metadata, TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY)).toEqual([]);
    },
  );

  it("reads only the metadata key it is asked for", () => {
    const metadata = {
      team_member_max_budget_alert_emails: { "50": ["team@example.com"] },
      max_budget_alert_emails: { "80": ["shared-mailbox@example.com"] },
    };
    expect(budgetAlertRowsFromMetadata(metadata, KEY_MAX_BUDGET_ALERT_EMAILS_KEY)).toEqual([
      { threshold: 80, emails: "shared-mailbox@example.com" },
    ]);
  });
});

describe("budgetAlertEmailsMetadata", () => {
  it("writes the threshold map under the given metadata key", () => {
    expect(
      budgetAlertEmailsMetadata(
        budgetAlertEmailsFromRows([{ threshold: 80, emails: "dl@example.com" }]),
        KEY_MAX_BUDGET_ALERT_EMAILS_KEY,
        {},
      ),
    ).toEqual({ max_budget_alert_emails: { "80": ["dl@example.com"] } });
  });

  it.each([{ rows: [] }, { rows: [{ threshold: null, emails: "dl@example.com" }] }])(
    "leaves the key out when no row has a valid threshold $rows",
    ({ rows }) => {
      expect(budgetAlertEmailsMetadata(budgetAlertEmailsFromRows(rows), KEY_MAX_BUDGET_ALERT_EMAILS_KEY, {})).toEqual(
        {},
      );
    },
  );

  it("keeps the entries the rows cannot show and lets the rows own every threshold they can", () => {
    const boxMetadata = {
      max_budget_alert_emails: {
        "50": ["old@example.com"],
        "80": ["removed@example.com"],
        "90": null,
        "150": ["over@example.com"],
        "0": ["zero@example.com"],
        "12.5": ["fraction@example.com"],
        soon: "label@example.com",
      },
    };

    expect(
      budgetAlertEmailsMetadata(
        budgetAlertEmailsFromRows([{ threshold: 50, emails: "new@example.com" }]),
        KEY_MAX_BUDGET_ALERT_EMAILS_KEY,
        boxMetadata,
      ),
    ).toEqual({
      max_budget_alert_emails: {
        "50": ["new@example.com"],
        "150": ["over@example.com"],
        "0": ["zero@example.com"],
        "12.5": ["fraction@example.com"],
        soon: "label@example.com",
      },
    });
  });
});

describe("budgetAlertEmailsRowsCanShow", () => {
  it("rebuilds the stored map together with the entries the rows cannot show, values untouched", () => {
    const stored = {
      max_budget_alert_emails: {
        "50": "a@example.com, b@example.com",
        "75": null,
        "80": ["c@example.com", null],
        "150": ["over@example.com"],
      },
    };

    expect(
      budgetAlertEmailsMetadata(
        budgetAlertEmailsRowsCanShow(stored, KEY_MAX_BUDGET_ALERT_EMAILS_KEY),
        KEY_MAX_BUDGET_ALERT_EMAILS_KEY,
        stored,
      ),
    ).toEqual(stored);
    expect(budgetAlertEmailsRowsCanShow(stored, KEY_MAX_BUDGET_ALERT_EMAILS_KEY)).not.toHaveProperty("150");
  });
});

describe("budgetAlertEmailsFromRows", () => {
  it("builds the threshold map, splitting, trimming and deduplicating recipients", () => {
    expect(
      budgetAlertEmailsFromRows([
        { threshold: 50, emails: "" },
        { threshold: 100, emails: " finance@example.com,cto@example.com , finance@example.com, " },
      ]),
    ).toEqual({ "50": [], "100": ["finance@example.com", "cto@example.com"] });
  });

  it("skips rows without a valid threshold", () => {
    expect(
      budgetAlertEmailsFromRows([
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
    expect(
      budgetAlertEmailsFromRows(budgetAlertRowsFromMetadata(stored, TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY)),
    ).toEqual(stored.team_member_max_budget_alert_emails);
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
