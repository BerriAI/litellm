export const CALENDAR_BUDGET_DURATIONS = ["1mo", "12mo"] as const;
export type BudgetPeriod = "monthly" | "yearly" | "other";

export const isCalendarBudgetDuration = (duration: string | null | undefined): boolean =>
  CALENDAR_BUDGET_DURATIONS.some((calendar) => calendar === duration);

export function budgetFormWindows<T extends { budget_duration: string }>(
  windows: readonly T[] | null | undefined,
): T[] {
  const calendar = (windows ?? []).filter((window) => isCalendarBudgetDuration(window.budget_duration));
  return calendar.length ? calendar : (windows ?? []).slice(0, 1);
}

export function budgetLimitAmount(
  windows: readonly { budget_duration: string; max_budget: number }[],
  scalar: number | null | undefined,
): number | null {
  return (
    windows.find((window) => budgetPeriod(window.budget_duration) === "monthly")?.max_budget ??
    scalar ??
    windows.find((window) => budgetPeriod(window.budget_duration) === "yearly")?.max_budget ??
    windows[0]?.max_budget ??
    null
  );
}

export function budgetPeriod(duration: string | null): BudgetPeriod {
  const normalized = duration?.trim().toLowerCase() ?? "";
  if ([CALENDAR_BUDGET_DURATIONS[0], "1month", "1months", "30d", "30days", "monthly"].includes(normalized))
    return "monthly";
  if ([CALENDAR_BUDGET_DURATIONS[1], "12month", "12months"].includes(normalized)) return "yearly";
  return "other";
}

export function mergeBudgetLimits<T extends { budget_duration: string }>(
  saved: readonly T[] | null | undefined,
  edited: readonly T[],
  shownDurations: readonly string[],
): T[] {
  return [
    ...edited,
    ...(saved ?? []).filter(
      (window) =>
        !shownDurations.includes(window.budget_duration) &&
        !edited.some((replacement) => replacement.budget_duration === window.budget_duration),
    ),
  ];
}
