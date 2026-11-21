import type {
  TernaryDryRunResponse,
  TernaryExportResponse,
  TernarySettingsView,
} from "@/app/(dashboard)/hooks/ternary/useTernary";
import i18n from "@/i18n/i18n";

export const describeFrequency = ({ frequency, interval_seconds }: TernarySettingsView): string =>
  frequency === "interval"
    ? i18n.t("loggingAndAlerts.ternary.settings.frequencyEvery", { seconds: interval_seconds })
    : i18n.t("loggingAndAlerts.ternary.settings.frequencyDaily");

export const formatEstimatedSpend = (spend: number): string =>
  spend.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 4 });

export const describeExport = ({ days_uploaded, start_time_utc, end_time_utc }: TernaryExportResponse): string =>
  days_uploaded.length
    ? i18n.t("loggingAndAlerts.ternary.settings.exportUploaded", {
        dayCount: days_uploaded.length,
        days: days_uploaded.join(", "),
      })
    : i18n.t("loggingAndAlerts.ternary.settings.exportEmpty", { start: start_time_utc, end: end_time_utc });

export const describePreview = (summary: TernaryDryRunResponse["summary"]): string => {
  const values = {
    records: summary.total_records,
    days: summary.days.length,
    spend: formatEstimatedSpend(summary.estimated_spend),
    tokens: summary.total_tokens.toLocaleString(),
    teams: summary.unique_teams,
    models: summary.unique_models,
  };
  return i18n.t("loggingAndAlerts.ternary.settings.previewSummary", values);
};
