import type { TFunction } from "i18next";
import i18n from "@/i18n/i18n";
import {
  type BudgetAlertRow,
  budgetAlertEmailsRowsCannotShow,
  budgetAlertRowsFromMetadata,
  KEY_MAX_BUDGET_ALERT_EMAILS_KEY,
} from "../key_team_helpers/budgetAlertEmails";

const WORD_FORM_BUDGET_DURATIONS: Record<string, string> = {
  hourly: "1h",
  daily: "24h",
  weekly: "7d",
  monthly: "30d",
};

// Normalize any legacy word-form budget duration to the canonical value the dropdown uses
export const canonicalBudgetDuration = (duration: string | null | undefined): string | null =>
  duration ? WORD_FORM_BUDGET_DURATIONS[duration] ?? duration : null;

export const keyOrganizationId = (key: { organization_id?: string | null; org_id?: string | null }): string | null =>
  key.organization_id || key.org_id || null;

// Determine the key_type display value from allowed_routes
export const keyTypeFromRoutes = (allowedRoutes: string[] | null | undefined): string => {
  if (!allowedRoutes || allowedRoutes.length === 0) return "default";
  if (allowedRoutes.includes("llm_api_routes")) return "llm_api";
  if (allowedRoutes.includes("management_routes")) return "management";
  if (allowedRoutes.includes("info_routes")) return "read_only";
  return "default";
};

export const parseAllowedRoutes = (value: unknown): string[] =>
  typeof value === "string" && value.trim() !== ""
    ? value
        .split(",")
        .map((route) => route.trim())
        .filter((route) => route.length > 0)
    : [];

export const modelSentinelOptions = (
  keyTeamId: string | null | undefined,
  teamLoaded: boolean,
  t: TFunction = i18n.t.bind(i18n),
): { value: string; label: string }[] => {
  if (keyTeamId == null)
    return [{ value: "all-proxy-models", label: t("keyDetail.keyEditFieldNormalizers.allProxyModels") }];
  return teamLoaded ? [{ value: "all-team-models", label: t("keyDetail.keyEditFieldNormalizers.allTeamModels") }] : [];
};

export type MovedMetadataTags = {
  metadata: string;
  tags: string[];
  movedTags: string[];
};

const parseJsonObject = (text: string): Record<string, unknown> | null => {
  try {
    const parsed: unknown = JSON.parse(text);
    return parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)
      ? (parsed as Record<string, unknown>)
      : null;
  } catch {
    return null;
  }
};

export const moveTagsOutOfMetadataJson = (
  metadataJson: string | undefined,
  currentTags: readonly string[] | undefined,
): MovedMetadataTags | null => {
  const parsed = metadataJson === undefined ? null : parseJsonObject(metadataJson);
  if (parsed === null) return null;
  const { tags: metadataTags, ...rest } = parsed;
  if (!Array.isArray(metadataTags)) return null;
  const existing = currentTags ?? [];
  const movedTags = metadataTags
    .filter((tag: unknown): tag is string => typeof tag === "string")
    .map((tag) => tag.trim())
    .filter((tag, index, all) => tag.length > 0 && !existing.includes(tag) && all.indexOf(tag) === index);
  return { metadata: JSON.stringify(rest, null, 2), tags: [...existing, ...movedTags], movedTags };
};

export type MovedMetadataBudgetAlerts = {
  readonly metadata: string;
  readonly rows: readonly BudgetAlertRow[];
  readonly movedThresholds: readonly (number | null)[];
};

export const moveBudgetAlertEmailsOutOfMetadataJson = (
  metadataJson: string | undefined,
  currentRows: readonly BudgetAlertRow[] | undefined,
): MovedMetadataBudgetAlerts | null => {
  const parsed = metadataJson === undefined ? null : parseJsonObject(metadataJson);
  const typedRows = budgetAlertRowsFromMetadata(parsed, KEY_MAX_BUDGET_ALERT_EMAILS_KEY);
  if (parsed === null || typedRows.length === 0) return null;
  const movedThresholds = typedRows.map((row) => row.threshold);
  const unmoved = budgetAlertEmailsRowsCannotShow(parsed, KEY_MAX_BUDGET_ALERT_EMAILS_KEY);
  const { [KEY_MAX_BUDGET_ALERT_EMAILS_KEY]: _moved, ...rest } = parsed;
  const metadata = Object.keys(unmoved).length > 0 ? { ...rest, [KEY_MAX_BUDGET_ALERT_EMAILS_KEY]: unmoved } : rest;
  const keptRows = (currentRows ?? []).filter((row) => !movedThresholds.includes(row.threshold));
  return { metadata: JSON.stringify(metadata, null, 2), rows: [...keptRows, ...typedRows], movedThresholds };
};

export const currentValuePlaceholder = (
  premiumUser: boolean,
  current: unknown,
  premiumHint: string,
  emptyHint: string,
): string => {
  if (!premiumUser) return premiumHint;
  return Array.isArray(current) && current.length > 0
    ? i18n.t("keyDetail.keyEditFieldNormalizers.currentValue", { values: current.join(", ") })
    : emptyHint;
};
