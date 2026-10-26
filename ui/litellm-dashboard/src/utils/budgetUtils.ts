import dayjs from "dayjs";
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
  tooltip: string;
}

const ROLLOVER_NOTE = "Spend above a rollover cap at reset carries into the next cycle.";

export function describeBudgetResetSchedule(
  budgetDuration: string | null | undefined,
  budgetResetAt: string | null | undefined,
): BudgetResetSchedule {
  const resetDate = budgetResetAt ? dayjs(budgetResetAt) : null;
  if (resetDate?.isValid()) {
    const cadence = budgetDuration ? `every ${budgetDuration}` : "on a schedule";
    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const label = resetDate.format("MMM D, YYYY HH:mm");
    return {
      label,
      resetAt: resetDate.toISOString(),
      tooltip: `Current Cycle Spend resets ${cadence}. Next reset ${label} (${timeZone}). ${ROLLOVER_NOTE}`,
    };
  }
  if (!budgetDuration) {
    return {
      label: "Never resets",
      resetAt: null,
      tooltip:
        "This budget has no reset window, so Current Cycle Spend accumulates for as long as the membership exists.",
    };
  }
  return {
    label: "Unknown",
    resetAt: null,
    tooltip: `Resets every ${budgetDuration}, but the next reset time has not been scheduled yet. ${ROLLOVER_NOTE}`,
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

export function budgetPeriodRange(
  budgetDuration: string | null | undefined,
  budgetResetAt: string | null | undefined,
): BudgetPeriod | null {
  if (!budgetDuration || !budgetResetAt) return null;
  const end = dayjs(budgetResetAt);
  if (!end.isValid()) return null;
  const match = /^(\d+)(mo|s|m|h|d|w)$/.exec(budgetDuration.trim());
  if (!match) return null;
  const amount = Number(match[1]);
  if (amount <= 0) return null;
  return { start: end.subtract(amount, BUDGET_DURATION_UNITS[match[2]]).toISOString(), end: end.toISOString() };
}
