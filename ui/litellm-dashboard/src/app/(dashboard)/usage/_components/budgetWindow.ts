import moment from "moment";

const budgetDurationAliases: Readonly<Record<string, string>> = {
  hourly: "1h",
  daily: "24h",
  weekly: "7d",
  monthly: "30d",
};

export const lastBudgetResetAt = (
  budgetResetAt: string | null | undefined,
  budgetDuration: string | null | undefined,
): Date | null => {
  if (!budgetResetAt || !budgetDuration) return null;

  const resetAt = moment(budgetResetAt);
  if (!resetAt.isValid()) return null;

  const normalizedDuration = budgetDuration.trim().toLowerCase();
  const duration = budgetDurationAliases[normalizedDuration] ?? normalizedDuration;
  const match = /^(\d+)(mo|[smhdw])$/.exec(duration);
  if (!match) return null;

  const value = Number(match[1]);
  const unit = match[2];

  if (unit === "mo" || (unit === "d" && value === 30)) {
    return resetAt.subtract(1, "month").toDate();
  }

  if (unit === "w") {
    return resetAt.subtract(value * 7, "days").toDate();
  }

  if (unit === "d") {
    return resetAt.subtract(value, "days").toDate();
  }

  if (unit === "h") {
    return resetAt.subtract(value, "hours").toDate();
  }

  if (unit === "m") {
    return resetAt.subtract(value, "minutes").toDate();
  }

  return resetAt.subtract(value, "seconds").toDate();
};
