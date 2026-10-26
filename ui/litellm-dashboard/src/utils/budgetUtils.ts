import dayjs, { Dayjs } from "dayjs";
import utc from "dayjs/plugin/utc";

dayjs.extend(utc);

export function formatBudgetReset(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const resetDate = dayjs(iso);
  if (!resetDate.isValid()) return null;
  return resetDate.format("MMM D, YYYY");
}

export function formatBudgetResetUtc(iso: string | null | undefined): string | null {
  if (!iso) return null;
  const resetDate = dayjs.utc(iso);
  if (!resetDate.isValid()) return null;
  return resetDate.format("MMM D, YYYY");
}

export interface BudgetResetSchedule {
  label: string;
  resetAt: string | null;
  period: string | null;
  tooltip: string;
}

const ROLLOVER_NOTE = "Spend above a rollover cap at reset carries into the next period.";

const DURATION_WORD_ALIASES: Record<string, string> = {
  hourly: "1h",
  daily: "24h",
  weekly: "7d",
  monthly: "30d",
};

const DURATION_PATTERN = /^(\d+)(mo|[smhdw])$/;

// Mirrors the proxy's get_budget_window_start: "30d" and "Nmo" span calendar months,
// everything else is a fixed length, and an unrecognised duration resets at the next midnight.
export function getBudgetPeriodStart(budgetDuration: string, resetAt: Dayjs): Dayjs {
  const trimmed = budgetDuration.trim();
  const normalized = DURATION_WORD_ALIASES[trimmed.toLowerCase()] ?? trimmed;
  const match = DURATION_PATTERN.exec(normalized);
  if (!match) return resetAt.subtract(1, "day");
  const value = Number(match[1]);
  switch (match[2]) {
    case "mo":
      return resetAt.subtract(value, "month");
    case "d":
      return value === 30 ? resetAt.subtract(1, "month") : resetAt.subtract(value, "day");
    case "w":
      return resetAt.subtract(value, "week");
    case "h":
      return resetAt.subtract(value, "hour");
    case "m":
      return resetAt.subtract(value, "minute");
    default:
      return resetAt.subtract(value, "second");
  }
}

export function formatBudgetPeriod(start: Dayjs, end: Dayjs): string {
  if (start.isSame(end, "day")) return `${start.format("MMM D, YYYY HH:mm")} - ${end.format("HH:mm")}`;
  if (start.isSame(end, "year")) return `${start.format("MMM D")} - ${end.format("MMM D, YYYY")}`;
  return `${start.format("MMM D, YYYY")} - ${end.format("MMM D, YYYY")}`;
}

export function describeBudgetResetSchedule(
  budgetDuration: string | null | undefined,
  budgetResetAt: string | null | undefined,
): BudgetResetSchedule {
  const resetDate = budgetResetAt ? dayjs(budgetResetAt) : null;
  if (resetDate?.isValid()) {
    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const label = resetDate.format("MMM D, YYYY HH:mm");
    const period = budgetDuration
      ? formatBudgetPeriod(getBudgetPeriodStart(budgetDuration, resetDate), resetDate)
      : null;
    const coverage = period
      ? `Current Period Spend covers ${period} and resets every ${budgetDuration}.`
      : "Current Period Spend resets on a schedule.";
    return {
      label,
      resetAt: resetDate.toISOString(),
      period,
      tooltip: `${coverage} Next reset ${label} (${timeZone}). ${ROLLOVER_NOTE}`,
    };
  }
  if (!budgetDuration) {
    return {
      label: "Never resets",
      resetAt: null,
      period: null,
      tooltip: "This budget has no reset window, so this is lifetime spend for as long as the membership exists.",
    };
  }
  return {
    label: "Unknown",
    resetAt: null,
    period: null,
    tooltip: `Resets every ${budgetDuration}, but the next reset time has not been scheduled yet. ${ROLLOVER_NOTE}`,
  };
}
