import type { TernaryExportResponse, TernarySettingsView } from "@/app/(dashboard)/hooks/ternary/useTernary";

export const describeFrequency = ({ frequency, interval_seconds }: TernarySettingsView): string =>
  frequency === "interval" ? `Every ${interval_seconds} seconds` : "Daily";

export const formatEstimatedSpend = (spend: number): string =>
  spend.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 4 });

export const describeExport = ({ days_uploaded, start_time_utc, end_time_utc }: TernaryExportResponse): string =>
  days_uploaded.length
    ? `Uploaded ${days_uploaded.length} day(s), one file each: ${days_uploaded.join(", ")} (UTC).`
    : `No usage between ${start_time_utc} and ${end_time_utc} (UTC). Nothing was sent to Ternary.`;
