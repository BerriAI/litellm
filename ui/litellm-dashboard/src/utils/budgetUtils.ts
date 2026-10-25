import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import { englishTranslate, type Translate } from "@/i18n/language";

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
  t: Translate = englishTranslate,
): BudgetResetSchedule {
  // 周期、时间和时区是业务值，只翻译包围它们的固定说明。
  const resetDate = budgetResetAt ? dayjs(budgetResetAt) : null;
  if (resetDate?.isValid()) {
    const cadence = budgetDuration ? t("every {duration}", { duration: budgetDuration }) : t("on a schedule");
    const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
    const label = resetDate.format("MMM D, YYYY HH:mm");
    const resetDetails = { cadence, label, timeZone, rollover: t(ROLLOVER_NOTE) };
    return {
      label,
      resetAt: resetDate.toISOString(),
      tooltip: t("Current Cycle Spend resets {cadence}. Next reset {label} ({timeZone}). {rollover}", resetDetails),
    };
  }
  if (!budgetDuration) {
    return {
      label: t("Never resets"),
      resetAt: null,
      tooltip: t(
        "This budget has no reset window, so Current Cycle Spend accumulates for as long as the membership exists.",
      ),
    };
  }
  return {
    label: t("Unknown"),
    resetAt: null,
    tooltip: t("Resets every {duration}, but the next reset time has not been scheduled yet. {rollover}", {
      duration: budgetDuration,
      rollover: t(ROLLOVER_NOTE),
    }),
  };
}
