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
  const windowSeconds = end.diff(start, "second");
  if (windowSeconds < 24 * 60 * 60) {
    const time = windowSeconds < 60 ? "HH:mm:ss" : "HH:mm";
    const endFormat = start.isSame(end, "day") ? time : `MMM D, YYYY ${time}`;
    return `${start.format(`MMM D, YYYY ${time}`)} - ${end.format(endFormat)}`;
  }
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
      ? formatBudgetPeriod(getBudgetPeriodStart(budgetDuration, resetDate.utc()).local(), resetDate)
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

const BUDGET_DURATION_WORD_ALIASES: Record<string, string> = {
  hourly: "1h",
  daily: "24h",
  weekly: "7d",
  monthly: "30d",
};

const BUDGET_DURATION_UNITS: Record<string, dayjs.ManipulateType> = {
  s: "second",
  m: "minute",
  h: "hour",
  d: "day",
  w: "week",
  mo: "month",
};

export interface BudgetPeriod {
  start: string;
  end: string;
}

/** Mirrors the proxy's get_budget_window_start: `30d` and `Nmo` span calendar months, the rest are fixed deltas. */
export function budgetPeriodRange(
  budgetDuration: string | null | undefined,
  budgetResetAt: string | null | undefined,
): BudgetPeriod | null {
  if (!budgetDuration || !budgetResetAt) return null;
  const end = dayjs.utc(budgetResetAt);
  if (!end.isValid()) return null;
  const normalized = budgetDuration.trim().toLowerCase();
  const match = /^(\d+)(mo|s|m|h|d|w)$/.exec(BUDGET_DURATION_WORD_ALIASES[normalized] ?? normalized);
  if (!match) return null;
  const amount = Number(match[1]);
  if (amount <= 0) return null;
  const start =
    match[2] === "d" && amount === 30
      ? end.subtract(1, "month")
      : end.subtract(amount, BUDGET_DURATION_UNITS[match[2]]);
  return { start: start.toISOString(), end: end.toISOString() };
}
