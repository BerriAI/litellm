export const NO_RESET = "never";

export const BUDGET_DURATION_OPTIONS = [
  { value: NO_RESET, label: "No reset" },
  { value: "1h", label: "hourly" },
  { value: "24h", label: "daily" },
  { value: "7d", label: "weekly" },
  { value: "30d", label: "monthly" },
] as const;
