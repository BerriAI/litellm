"use client";

import dayjs from "dayjs";
import { CircleHelp } from "lucide-react";

import { getBudgetDurationLabel } from "@/components/common_components/budget_duration_dropdown";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { budgetPeriodRange } from "@/utils/budgetUtils";
import { formatNumberWithCommas } from "@/utils/dataUtils";

export interface BudgetPeriodTooltipProps {
  spend: number | null | undefined;
  maxBudget: number | null | undefined;
  budgetDuration: string | null | undefined;
  budgetResetAt: string | null | undefined;
}

const formatPeriodDate = (iso: string): string => dayjs(iso).format("MMM D, YYYY HH:mm");

const describePeriod = (budgetDuration: string, budgetResetAt: string | null | undefined): string | null => {
  const timeZone = Intl.DateTimeFormat().resolvedOptions().timeZone;
  const period = budgetPeriodRange(budgetDuration, budgetResetAt);
  if (period) {
    return `Budget period: ${formatPeriodDate(period.start)} to ${formatPeriodDate(period.end)} (${timeZone})`;
  }
  if (budgetResetAt && dayjs(budgetResetAt).isValid()) {
    return `Next reset: ${formatPeriodDate(budgetResetAt)} (${timeZone})`;
  }
  return null;
};

export const budgetPeriodLines = ({
  spend,
  maxBudget,
  budgetDuration,
  budgetResetAt,
}: BudgetPeriodTooltipProps): readonly string[] => {
  const spendLine = `${budgetDuration ? "Spend this period" : "Spend so far"}: $${formatNumberWithCommas(spend ?? 0, 4)}`;
  if (maxBudget == null) return ["No budget set", spendLine];
  const maxBudgetLine = `Max budget: $${formatNumberWithCommas(maxBudget, 2)}`;
  if (!budgetDuration) return ["Lifetime budget, never resets", maxBudgetLine, spendLine];
  const periodLine = describePeriod(budgetDuration, budgetResetAt);
  return [
    ...(periodLine ? [periodLine] : []),
    maxBudgetLine,
    `Reset interval: ${getBudgetDurationLabel(budgetDuration)}`,
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
