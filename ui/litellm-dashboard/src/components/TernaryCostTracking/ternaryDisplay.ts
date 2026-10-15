import type { TernarySettingsView } from "@/app/(dashboard)/hooks/ternary/useTernary";

export const describeFrequency = ({ frequency, interval_seconds }: TernarySettingsView): string =>
  frequency === "interval" ? `Every ${interval_seconds} seconds` : "Daily";

export const formatEstimatedSpend = (spend: number): string =>
  spend.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 4 });
