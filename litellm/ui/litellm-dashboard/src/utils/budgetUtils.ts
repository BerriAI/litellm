import dayjs, { Dayjs } from "dayjs";
import utc from "dayjs/plugin/utc";
import type { TFunction } from "i18next";

import i18n from "@/i18n/i18n";
import { formatDateWithEnglishPattern } from "@/i18n/formats";

dayjs.extend(utc);

const UTC_MONTH_DAY = { month: "short", day: "numeric", timeZone: "UTC" } as const;
const UTC_DATE = { ...UTC_MONTH_DAY, year: "numeric" } as const;
const UTC_CLOCK = { hour: "2-digit", minute: "2-digit", hourCycle: "h23", timeZone: "UTC" } as const;
const UTC_TIMESTAMP = { ...UTC_DATE, ...UTC_CLOCK } as const;
const UTC_SECONDS = { second: "2-digit" } as const;
const UTC_12_HOUR_CLOCK = { ...UTC_CLOCK, hour: "numeric", hour12: true } as const;

const displayBudgetDate = (date: Dayjs, language: string, pattern: string, options: Intl.DateTimeFormatOptions) =>
  formatDateWithEnglishPattern(date.toDate(), language, date.format(pattern), options);

export const formatBudgetTimestamp = (date: Dayjs, language = i18n.language): string =>
  displayBudgetDate(date, language, "MMM D, YYYY HH:mm", UTC_TIMESTAMP);

export function formatBudgetResetTime(
  iso: string | null | undefined,
  t: TFunction = i18n.t.bind(i18n),
  language = i18n.language,
): string | null {
  if (!iso) return null;
  const resetDate = dayjs.utc(iso);
  if (!resetDate.isValid()) return null;
  const timestamp = t("keyDetail.keyInfoView.timestamp", {
    date: displayBudgetDate(resetDate, language, "MMM D, YYYY", UTC_DATE),
    time: displayBudgetDate(resetDate, language, "h:mm A", UTC_12_HOUR_CLOCK),
  });
  return `${timestamp} UTC`;
}

const isMidnight = (instant: Dayjs): boolean => instant.isSame(instant.startOf("day"));

export function formatBudgetResetUtc(iso: string | null | undefined, language = i18n.language): string | null {
  if (!iso) return null;
  const resetDate = dayjs.utc(iso);
  if (!resetDate.isValid()) return null;
  return isMidnight(resetDate)
    ? displayBudgetDate(resetDate, language, "MMM D, YYYY", UTC_DATE)
    : formatBudgetTimestamp(resetDate, language);
}

export interface BudgetResetSchedule {
  label: string;
  resetAt: string | null;
  period: string | null;
  tooltip: string;
}

const DURATION_WORD_ALIASES: Record<string, string> = {
  hourly: "1h",
  daily: "24h",
  weekly: "7d",
  monthly: "30d",
};

const DURATION_PATTERN = /^(\d+)(mo|[smhdw])$/;

export function normalizeBudgetDuration(duration: string): string {
  const trimmed = duration.trim();
  const normalized = DURATION_WORD_ALIASES[trimmed.toLowerCase()] ?? trimmed;
  const match = DURATION_PATTERN.exec(normalized);
  if (!match) return normalized;
  const amount = Number(match[1]);
  if (match[2] === "w") return `${amount * 7}d`;
  if (match[2] === "d" && amount === 30) return "1mo";
  return `${amount}${match[2]}`;
}

export function isValidBudgetDuration(duration: string): boolean {
  const match = DURATION_PATTERN.exec(normalizeBudgetDuration(duration));
  if (!match) return false;
  const amount = Number(match[1]);
  if (!Number.isSafeInteger(amount) || amount <= 0) return false;
  return match[2] !== "mo" || amount === 1 || amount === 12;
}

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

export function formatBudgetPeriod(start: Dayjs, end: Dayjs, language = i18n.language): string {
  const windowSeconds = end.diff(start, "second");
  if (windowSeconds < 24 * 60 * 60 || !isMidnight(start) || !isMidnight(end)) {
    const hasSeconds = windowSeconds < 60;
    const time = hasSeconds ? "HH:mm:ss" : "HH:mm";
    const clock = hasSeconds ? { ...UTC_CLOCK, ...UTC_SECONDS } : UTC_CLOCK;
    const timestamp = { ...UTC_DATE, ...clock };
    const pattern = `MMM D, YYYY ${time}`;
    const first = displayBudgetDate(start, language, pattern, timestamp);
    const last = start.isSame(end, "day")
      ? displayBudgetDate(end, language, time, clock)
      : displayBudgetDate(end, language, pattern, timestamp);
    return `${first} - ${last}`;
  }
  const first = start.isSame(end, "year")
    ? displayBudgetDate(start, language, "MMM D", UTC_MONTH_DAY)
    : displayBudgetDate(start, language, "MMM D, YYYY", UTC_DATE);
  return `${first} - ${displayBudgetDate(end, language, "MMM D, YYYY", UTC_DATE)}`;
}

export function describeBudgetResetSchedule(
  budgetDuration: string | null | undefined,
  budgetResetAt: string | null | undefined,
  t: TFunction = i18n.t.bind(i18n),
  language = i18n.language,
): BudgetResetSchedule {
  const resetDate = budgetResetAt ? dayjs.utc(budgetResetAt) : null;
  if (resetDate?.isValid()) {
    const timestamp = formatBudgetTimestamp(resetDate, language);
    const period = budgetDuration
      ? `${formatBudgetPeriod(getBudgetPeriodStart(budgetDuration, resetDate), resetDate, language)} UTC`
      : null;
    const coverage = period
      ? t("budgetSchedule.coverageWithPeriod", { period, duration: budgetDuration })
      : t("budgetSchedule.coverageWithoutPeriod");
    const tooltipValues = {
      coverage,
      label: timestamp,
      timeZone: "UTC",
      rolloverNote: t("budgetSchedule.rolloverNote"),
    };
    return {
      label: `${timestamp} UTC`,
      resetAt: resetDate.toISOString(),
      period,
      tooltip: t("budgetSchedule.scheduledTooltip", tooltipValues),
    };
  }
  if (!budgetDuration) {
    return {
      label: t("budgetSchedule.neverResets.label"),
      resetAt: null,
      period: null,
      tooltip: t("budgetSchedule.neverResets.tooltip"),
    };
  }
  return {
    label: t("budgetSchedule.unscheduled.label"),
    resetAt: null,
    period: null,
    tooltip: t("budgetSchedule.unscheduled.tooltip", {
      duration: budgetDuration,
      rolloverNote: t("budgetSchedule.rolloverNote"),
    }),
  };
}

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
  const match = DURATION_PATTERN.exec(DURATION_WORD_ALIASES[normalized] ?? normalized);
  if (!match) return null;
  const amount = Number(match[1]);
  if (amount <= 0) return null;
  const start =
    match[2] === "d" && amount === 30
      ? end.subtract(1, "month")
      : end.subtract(amount, BUDGET_DURATION_UNITS[match[2]]);
  return { start: start.toISOString(), end: end.toISOString() };
}
