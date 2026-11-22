import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { formatUsd } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { TableCell, TableHead } from "@/components/ui/table";
import { CustomerRateLimitSummary } from "./CustomerRateLimits";
import { customerBudgetLimitLabel, customerBudgetWindows } from "./customerBudgetUsage";
import type { CustomerUsageEntity } from "./customerUsageApi";

export function CustomerBudgetSortHeader({
  sort,
  onSort,
}: {
  sort: "budget_asc" | "budget_desc" | null;
  onSort: () => void;
}) {
  const { t } = useTranslation();
  const SortIcon = sort === "budget_desc" ? ArrowDown : ArrowUp;
  const Icon = sort ? SortIcon : ArrowUpDown;
  const direction = sort === "budget_desc" ? "descending" : "ascending";
  return (
    <TableHead className="px-4" aria-sort={sort ? direction : "none"}>
      <Button variant="ghost" size="sm" className="-mx-2" onClick={onSort}>
        {t("customerPortal.hierarchy.maxBudget")}
        <Icon aria-hidden="true" className="size-3.5" />
      </Button>
    </TableHead>
  );
}

export function CustomerBudgetLimitCell({ entity, children }: { entity: CustomerUsageEntity; children?: ReactNode }) {
  const { t } = useTranslation();
  const budgets = customerBudgetWindows(entity);
  return (
    <TableCell className="px-4 tabular-nums">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <div className="space-y-1">
          {budgets.length === 0 && <span>{t("customerPortal.rateLimits.noSpendLimit")}</span>}
          {budgets.map((budget) => (
            <div key={`${budget.source}-${budget.budget_duration}`} className="flex flex-col items-start gap-1">
              <p>{customerBudgetLimitLabel(formatUsd(budget.max_budget), budget.budget_duration, t)}</p>
              {!budget.budget_duration && (
                <Badge variant="secondary" className="font-normal text-muted-foreground">
                  {t("customerPortal.analytics.noAutomaticReset")}
                </Badge>
              )}
            </div>
          ))}
        </div>
        {children}
      </div>
      <CustomerRateLimitSummary limits={entity.rateLimits} />
    </TableCell>
  );
}
