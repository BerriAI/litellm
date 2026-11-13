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
  if (value === null) return [];
  if (typeof value === "string") return splitEmails(value);
  if (Array.isArray(value)) return value.filter((email): email is string => typeof email === "string");
  return null;
};

const THRESHOLD_MIN = 1;
const THRESHOLD_MAX = 100;

export const isValidThreshold = (threshold: number | null): threshold is number => {
  const isWholeNumber = threshold !== null && Number.isInteger(threshold);
  return isWholeNumber && threshold >= THRESHOLD_MIN && threshold <= THRESHOLD_MAX;
};

const isBudgetAlertMap = (value: unknown): value is Readonly<Record<string, unknown>> =>
  typeof value === "object" && value !== null && !Array.isArray(value);

const storedBudgetAlertValue = (metadata: unknown, metadataKey: BudgetAlertEmailsMetadataKey): unknown =>
  typeof metadata === "object" && metadata !== null ? (metadata as Record<string, unknown>)[metadataKey] : undefined;

const storedBudgetAlertConfig = (
  metadata: unknown,
  metadataKey: BudgetAlertEmailsMetadataKey,
): Readonly<Record<string, unknown>> => {
  const config = storedBudgetAlertValue(metadata, metadataKey);
  return isBudgetAlertMap(config) ? config : {};
};

const ROW_THRESHOLD_KEY_PATTERN: Readonly<Record<BudgetAlertEmailsMetadataKey, RegExp>> = {
  [KEY_MAX_BUDGET_ALERT_EMAILS_KEY]: /^[1-9]\d*$/,
  [TEAM_MEMBER_MAX_BUDGET_ALERT_EMAILS_KEY]: /^\d+$/,
};

const budgetAlertRowFromEntry = (
  [key, emails]: readonly [string, unknown],
  metadataKey: BudgetAlertEmailsMetadataKey,
): BudgetAlertRow | null => {
  const threshold = Number(key);
  const recipients = storedRecipients(emails);
  return ROW_THRESHOLD_KEY_PATTERN[metadataKey].test(key) && isValidThreshold(threshold) && recipients !== null
    ? { threshold, emails: recipients.join(", ") }
    : null;
};

export const budgetAlertRowsFromMetadata = (
  metadata: unknown,
  metadataKey: BudgetAlertEmailsMetadataKey,
): readonly BudgetAlertRow[] =>
  Object.entries(storedBudgetAlertConfig(metadata, metadataKey))
    .flatMap((entry) => {
      const row = budgetAlertRowFromEntry(entry, metadataKey);
      return row === null ? [] : [row];
    })
    .sort((a, b) => (a.threshold ?? 0) - (b.threshold ?? 0));

const storedBudgetAlertEntries = (
  metadata: unknown,
  metadataKey: BudgetAlertEmailsMetadataKey,
  rowsCanShow: boolean,
): Readonly<Record<string, unknown>> =>
  Object.fromEntries(
    Object.entries(storedBudgetAlertConfig(metadata, metadataKey)).filter(
      (entry) => (budgetAlertRowFromEntry(entry, metadataKey) !== null) === rowsCanShow,
    ),
  );

export const budgetAlertEmailsRowsCanShow = (
  metadata: unknown,
  metadataKey: BudgetAlertEmailsMetadataKey,
): Readonly<Record<string, unknown>> => storedBudgetAlertEntries(metadata, metadataKey, true);

export const budgetAlertEmailsRowsCannotShow = (
  metadata: unknown,
  metadataKey: BudgetAlertEmailsMetadataKey,
): Readonly<Record<string, unknown>> => storedBudgetAlertEntries(metadata, metadataKey, false);

export const budgetAlertEmailsFromRows = (rows: readonly BudgetAlertRow[]): BudgetAlertEmails =>
  Object.fromEntries(
    rows
      .filter((row) => isValidThreshold(row.threshold))
      .map((row) => [String(row.threshold), splitEmails(row.emails)]),
  );

export const budgetAlertEmailsMetadata = (
  rowEntries: Readonly<Record<string, unknown>>,
  metadataKey: BudgetAlertEmailsMetadataKey,
  metadataWithHiddenEntries: unknown,
): Readonly<Partial<Record<BudgetAlertEmailsMetadataKey, unknown>>> => {
  const boxValue = storedBudgetAlertValue(metadataWithHiddenEntries, metadataKey);
  if (Object.keys(rowEntries).length === 0 && boxValue !== undefined && !isBudgetAlertMap(boxValue)) {
    return { [metadataKey]: boxValue };
  }
  const emails = {
    ...budgetAlertEmailsRowsCannotShow(metadataWithHiddenEntries, metadataKey),
    ...rowEntries,
  };
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
