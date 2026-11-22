"use client";

import { useTranslation } from "react-i18next";

import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { SCOPE_LABELS } from "@/components/shared/InheritedBudgetHint";
import { Meter, MeterIndicator, MeterTrack } from "@/components/shared/Meter";
import { getSpendBudgetMeterTone } from "@/components/shared/table_cells/spend_budget_cell";
import { budgetUsagePercent, closestToLimit, type AppliedBudget } from "@/components/shared/appliedBudgets";
import { cn } from "@/lib/cva.config";
import { formatNumberWithCommas } from "@/utils/dataUtils";

export function AppliedBudgetsCard({ budgets, className }: { budgets: readonly AppliedBudget[]; className?: string }) {
  const { t } = useTranslation();
  const closest = closestToLimit(budgets);

  return (
    <Card className={cn("block p-6", className)} data-testid="applied-budgets">
      <p className="text-sm font-medium">{t("shared.appliedBudgets.title")}</p>
      <p className="mt-1 text-sm text-muted-foreground">
        {budgets.length === 0 ? t("shared.appliedBudgets.empty") : t("shared.appliedBudgets.explainer")}
      </p>
      {budgets.length > 0 && (
        <ul className="mt-4 divide-y">
          {budgets.map((budget) => {
            const usage = budgetUsagePercent(budget);
            const limit = formatNumberWithCommas(budget.maxBudget, 2);
            return (
              <li key={budget.scope} className="py-3 first:pt-0 last:pb-0">
                <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 text-sm">
                  <div className="flex min-w-0 items-center gap-2">
                    <span className="shrink-0 text-muted-foreground">{t(SCOPE_LABELS[budget.scope])}</span>
                    <span className="truncate font-medium">{budget.name}</span>
                    {budget === closest && <Badge variant="outline">{t("shared.appliedBudgets.closestToLimit")}</Badge>}
                  </div>
                  <span className="whitespace-nowrap tabular-nums">
                    {budget.spend == null
                      ? t("shared.appliedBudgets.limitOnly", { limit })
                      : t("shared.appliedBudgets.spendOfLimit", {
                          spend: formatNumberWithCommas(budget.spend, 2),
                          limit,
                        })}
                  </span>
                </div>
                {usage !== null && (
                  <Meter
                    className="mt-2"
                    value={Math.min(usage, 100)}
                    max={100}
                    aria-label={t(SCOPE_LABELS[budget.scope])}
                    aria-valuetext={t("shared.appliedBudgets.percentUsed", { percent: Math.round(usage) })}
                  >
                    <MeterTrack>
                      <MeterIndicator tone={getSpendBudgetMeterTone(usage)} />
                    </MeterTrack>
                  </Meter>
                )}
                <p className="mt-1 text-xs text-muted-foreground">
                  {budget.budgetDuration
                    ? t("shared.appliedBudgets.resetsEvery", { duration: budget.budgetDuration })
                    : t("shared.appliedBudgets.noReset")}
                </p>
              </li>
            );
          })}
        </ul>
      )}
    </Card>
  );
}
