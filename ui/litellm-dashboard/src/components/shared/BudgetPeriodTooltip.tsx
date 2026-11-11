"use client";

import dayjs from "dayjs";
import utc from "dayjs/plugin/utc";
import { CircleHelp } from "lucide-react";

import { getBudgetDurationLabel } from "@/components/common_components/budget_duration_dropdown";
import i18n from "@/i18n/i18n";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { budgetPeriodRange } from "@/utils/budgetUtils";
import { formatNumberWithCommas } from "@/utils/dataUtils";

export interface BudgetPeriodTooltipProps {
  spend: number | null | undefined;
  maxBudget: number | null | undefined;
  budgetDuration: string | null | undefined;
  budgetResetAt: string | null | undefined;
}

dayjs.extend(utc);

const formatPeriodDate = (iso: string): string => dayjs.utc(iso).format("MMM D, YYYY HH:mm");

const describePeriod = (budgetDuration: string, budgetResetAt: string | null | undefined): string | null => {
  const period = budgetPeriodRange(budgetDuration, budgetResetAt);
  if (period) {
    return i18n.t("shared.budgetPeriodTooltip.budgetPeriod", {
      start: formatPeriodDate(period.start),
      end: formatPeriodDate(period.end),
      timeZone: "UTC",
    });
  }
  if (budgetResetAt && dayjs.utc(budgetResetAt).isValid()) {
    return i18n.t("shared.budgetPeriodTooltip.nextReset", { date: formatPeriodDate(budgetResetAt), timeZone: "UTC" });
  }
  return null;
};

export const budgetPeriodLines = ({
  spend,
  maxBudget,
  budgetDuration,
  budgetResetAt,
}: BudgetPeriodTooltipProps): readonly string[] => {
  const spendLine = i18n.t("shared.budgetPeriodTooltip.spendLine", {
    label: budgetDuration
      ? i18n.t("shared.budgetPeriodTooltip.spendThisPeriod")
      : i18n.t("shared.budgetPeriodTooltip.spendSoFar"),
    amount: formatNumberWithCommas(spend ?? 0, 4),
  });
  if (maxBudget == null) return [i18n.t("shared.budgetPeriodTooltip.noBudgetSet"), spendLine];
  const maxBudgetLine = i18n.t("shared.budgetPeriodTooltip.maxBudget", {
    amount: formatNumberWithCommas(maxBudget, 2),
  });
  if (!budgetDuration) return [i18n.t("shared.budgetPeriodTooltip.lifetimeBudget"), maxBudgetLine, spendLine];
  const periodLine = describePeriod(budgetDuration, budgetResetAt);
  return [
    ...(periodLine ? [periodLine] : []),
    maxBudgetLine,
    i18n.t("shared.budgetPeriodTooltip.resetInterval", { interval: getBudgetDurationLabel(budgetDuration) }),
    spendLine,
  ];
};

export function BudgetPeriodTooltip(props: BudgetPeriodTooltipProps) {
  return (
    <SimpleTooltip
      content={
        <div data-testid="budget-period-tooltip" className="flex flex-col gap-1">
          {budgetPeriodLines(props).map((line) => (
            <span key={line}>{line}</span>
          ))}
        </div>
      }
    >
      <CircleHelp aria-label="budget-period-details" className="ml-1 size-4 text-muted-foreground" />
    </SimpleTooltip>
  );
}
