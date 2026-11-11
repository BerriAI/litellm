import { z } from "zod";

export const TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY = "team_member_max_budget_alert_emails" as const;
export const KEY_MAX_BUDGET_ALERT_EMAILS_KEY = "max_budget_alert_emails" as const;

export type BudgetAlertEmailsMetadataKey =
  | typeof TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY
  | typeof KEY_MAX_BUDGET_ALERT_EMAILS_KEY;

export interface BudgetAlertRow {
  readonly threshold: number | null;
  readonly emails: string;
}

export type BudgetAlertEmails = Readonly<Record<string, readonly string[]>>;

export type BudgetAlertRowIssue = "thresholdRange" | "duplicateThreshold";

const splitEmails = (emails: string): readonly string[] =>
  Array.from(
    new Set(
      emails
        .split(",")
        .map((email) => email.trim())
        .filter((email) => email.length > 0),
    ),
  );

const storedRecipients = (value: unknown): readonly string[] | null => {
  if (typeof value === "string") return splitEmails(value);
  if (Array.isArray(value) && value.every((email) => typeof email === "string")) return value;
  return null;
};

const THRESHOLD_MIN = 1;
const THRESHOLD_MAX = 100;

export const isValidThreshold = (threshold: number | null): threshold is number => {
  const isWholeNumber = threshold !== null && Number.isInteger(threshold);
  return isWholeNumber && threshold >= THRESHOLD_MIN && threshold <= THRESHOLD_MAX;
};

export const budgetAlertRowsFromMetadata = (
  metadata: unknown,
  metadataKey: BudgetAlertEmailsMetadataKey,
): readonly BudgetAlertRow[] => {
  if (typeof metadata !== "object" || metadata === null) return [];
  const config: unknown = (metadata as Record<string, unknown>)[metadataKey];
  if (typeof config !== "object" || config === null || Array.isArray(config)) return [];
  return Object.entries(config as Record<string, unknown>)
    .flatMap(([key, emails]) => {
      const threshold = Number(key);
      const recipients = storedRecipients(emails);
      return /^\d+$/.test(key) && isValidThreshold(threshold) && recipients !== null
        ? [{ threshold, emails: recipients.join(", ") }]
        : [];
    })
    .sort((a, b) => (a.threshold ?? 0) - (b.threshold ?? 0));
};

export const budgetAlertEmailsFromRows = (rows: readonly BudgetAlertRow[]): BudgetAlertEmails =>
  Object.fromEntries(
    rows
      .filter((row) => isValidThreshold(row.threshold))
      .map((row) => [String(row.threshold), splitEmails(row.emails)]),
  );

export const budgetAlertEmailsMetadata = (
  rows: readonly BudgetAlertRow[] | undefined,
  metadataKey: BudgetAlertEmailsMetadataKey,
): Readonly<Partial<Record<BudgetAlertEmailsMetadataKey, BudgetAlertEmails>>> => {
  const emails = budgetAlertEmailsFromRows(rows ?? []);
  return Object.keys(emails).length > 0 ? { [metadataKey]: emails } : {};
};

export const budgetAlertRowsSchema = (message: (issue: BudgetAlertRowIssue) => string) =>
  z.array(z.object({ threshold: z.number().nullable(), emails: z.string() })).superRefine((rows, ctx) => {
    rows.forEach((row, index) => {
      if (!isValidThreshold(row.threshold)) {
        ctx.addIssue({ code: "custom", message: message("thresholdRange"), path: [index, "threshold"] });
      } else if (rows.filter((other) => other.threshold === row.threshold).length > 1) {
        ctx.addIssue({ code: "custom", message: message("duplicateThreshold"), path: [index, "threshold"] });
      }
    });
  });

export const teamMemberBudgetAlertSummary = (metadata: unknown): readonly string[] =>
  budgetAlertRowsFromMetadata(metadata, TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY).map((row) =>
    row.emails.length > 0 ? `${row.threshold}%: member, ${row.emails}` : `${row.threshold}%: member`,
  );
