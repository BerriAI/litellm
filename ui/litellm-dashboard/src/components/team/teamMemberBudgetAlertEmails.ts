export const TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY = "team_member_max_budget_alert_emails" as const;

export interface TeamMemberBudgetAlertRow {
  readonly threshold: number | null;
  readonly emails: string;
}

export type TeamMemberBudgetAlertEmails = Readonly<Record<string, readonly string[]>>;

const isEmailList = (value: unknown): value is readonly string[] =>
  Array.isArray(value) && value.every((email) => typeof email === "string");

const splitEmails = (emails: string): readonly string[] =>
  Array.from(
    new Set(
      emails
        .split(",")
        .map((email) => email.trim())
        .filter((email) => email.length > 0),
    ),
  );

const THRESHOLD_MIN = 1;
const THRESHOLD_MAX = 100;

export const isValidThreshold = (threshold: number | null): threshold is number => {
  const isWholeNumber = threshold !== null && Number.isInteger(threshold);
  return isWholeNumber && threshold >= THRESHOLD_MIN && threshold <= THRESHOLD_MAX;
};

export const teamMemberBudgetAlertRowsFromMetadata = (metadata: unknown): readonly TeamMemberBudgetAlertRow[] => {
  if (typeof metadata !== "object" || metadata === null) return [];
  const config: unknown = (metadata as Record<string, unknown>)[TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY];
  if (typeof config !== "object" || config === null || Array.isArray(config)) return [];
  return Object.entries(config as Record<string, unknown>)
    .flatMap(([key, emails]) => {
      const threshold = Number(key);
      return /^\d+$/.test(key) && isValidThreshold(threshold) && isEmailList(emails)
        ? [{ threshold, emails: emails.join(", ") }]
        : [];
    })
    .sort((a, b) => (a.threshold ?? 0) - (b.threshold ?? 0));
};

export const teamMemberBudgetAlertEmailsFromRows = (
  rows: readonly TeamMemberBudgetAlertRow[],
): TeamMemberBudgetAlertEmails =>
  Object.fromEntries(
    rows
      .filter((row) => isValidThreshold(row.threshold))
      .map((row) => [String(row.threshold), splitEmails(row.emails)]),
  );

export const teamMemberBudgetAlertSummary = (metadata: unknown): readonly string[] =>
  teamMemberBudgetAlertRowsFromMetadata(metadata).map((row) =>
    row.emails.length > 0 ? `${row.threshold}%: member, ${row.emails}` : `${row.threshold}%: member`,
  );
