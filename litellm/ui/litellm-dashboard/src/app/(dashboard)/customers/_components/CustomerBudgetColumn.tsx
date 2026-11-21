import { ArrowDown, ArrowUp, ArrowUpDown } from "lucide-react";
import type { ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { formatUsd } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { Button } from "@/components/ui/button";
import { TableCell, TableHead } from "@/components/ui/table";
import { CustomerRateLimitSummary, type CustomerRateLimits } from "./CustomerRateLimits";

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

export function CustomerBudgetLimitCell({
  limit,
  rateLimits,
  children,
}: {
  limit: number | null;
  rateLimits?: CustomerRateLimits | null;
  children?: ReactNode;
}) {
  const { t } = useTranslation();
  return (
    <TableCell className="px-4 tabular-nums">
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span>{limit === null ? t("customerPortal.analytics.noLimit") : formatUsd(limit)}</span>
        {children}
      </div>
      <CustomerRateLimitSummary limits={rateLimits} />
    </TableCell>
  );
}
