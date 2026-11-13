import { useTranslation } from "react-i18next";
import { formatCompact, formatUsd } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { TableCell, TableHead } from "@/components/ui/table";
import { cn } from "@/lib/cva.config";
import type { CustomerUsageEntity } from "./customerUsageApi";
import CustomerBudgetSpend, { CustomerBudgetWindow } from "./CustomerBudgetSpend";
import { customerBudgetPeriod, customerBudgetStatus, customerBudgetUsage } from "./customerBudgetUsage";

export function CustomerSpendIdentity({
  entity: member,
  onClick,
}: {
  entity: CustomerUsageEntity;
  onClick: () => void;
}) {
  return (
    <div className="flex items-center gap-3">
      <span
        aria-hidden="true"
        className="flex size-8 shrink-0 items-center justify-center rounded-full bg-muted text-xs font-medium"
      >
        {member.label
          .split(/\s+/)
          .slice(0, 2)
          .map((part) => part.charAt(0))
          .join("")
          .toLocaleUpperCase()}
      </span>
      <div className="min-w-0">
        <button
          type="button"
          className="block text-left font-medium hover:text-primary focus-visible:outline-2 focus-visible:outline-ring"
          onClick={onClick}
        >
          {member.label}
        </button>
        <span className="block font-mono text-[11px] text-muted-foreground">{member.id}</span>
      </div>
    </div>
  );
}

export function CustomerSpendHeaders() {
  const { t } = useTranslation();
  return (
    <>
      <TableHead className="min-w-72">{t("customerPortal.analytics.monthlySpend")}</TableHead>
      <TableHead className="text-right">{t("customerPortal.analytics.yearlySpend")}</TableHead>
      <TableHead className="text-right">{t("customerPortal.analytics.customerSpendShare")}</TableHead>
      <TableHead className="text-right">{t("customerPortal.analytics.tokens")}</TableHead>
      <TableHead className="pr-0 text-right">{t("customerPortal.analytics.status")}</TableHead>
    </>
  );
}

export function CustomerSpendCells({
  entity: member,
  monthlySpend,
  yearlySpend,
  tokens,
  organizationSpend,
}: {
  entity: CustomerUsageEntity;
  monthlySpend: number;
  yearlySpend: number;
  tokens: number;
  organizationSpend: number;
}) {
  const { t } = useTranslation();
  const budgets = customerBudgetUsage(member);
  const monthlyBudgets = budgets.filter((budget) => customerBudgetPeriod(budget.budget_duration) === "monthly");
  const yearlyBudgets = budgets.filter((budget) => customerBudgetPeriod(budget.budget_duration) === "yearly");
  const otherBudgets = budgets.filter((budget) => customerBudgetPeriod(budget.budget_duration) === "other");

  const spendShare = organizationSpend > 0 ? (monthlySpend / organizationSpend) * 100 : 0;
  const sharePercentage = Math.max(0, Math.min(100, spendShare));
  const shareLabel = t("format.number", {
    val: spendShare / 100,
    style: "percent",
    maximumFractionDigits: 1,
  });
  const status = customerBudgetStatus(budgets);
  const atLimit = status === "customerPortal.analytics.reachedLimit";

  return (
    <>
      <TableCell className="pr-6">
        <div className="space-y-3">
          {monthlyBudgets.map((budget) => (
            <CustomerBudgetWindow key={budget.source} budget={budget} showReset={monthlyBudgets.length > 1} />
          ))}
          {monthlyBudgets.length === 0 && (
            <CustomerBudgetSpend
              spend={monthlySpend}
              limit={null}
              label={t("customerPortal.analytics.monthlySpend")}
              noLimitLabel={t("customerPortal.analytics.noMonthlyLimit")}
            />
          )}
          {otherBudgets.map((budget) => (
            <div key={budget.source} className="space-y-1 text-xs text-muted-foreground">
              <span>
                {t("customerPortal.analytics.budgetPeriod")}
                {budget.budget_duration && ` (${budget.budget_duration})`}
              </span>
              <CustomerBudgetWindow budget={budget} />
            </div>
          ))}
        </div>
      </TableCell>
      <TableCell className="text-right tabular-nums">
        <div className="space-y-2">
          {yearlyBudgets.map((budget) => (
            <CustomerBudgetWindow key={budget.source} budget={budget} showReset={yearlyBudgets.length > 1} />
          ))}
          {yearlyBudgets.length === 0 && `${formatUsd(yearlySpend)} / ${t("customerPortal.analytics.noYearlyLimit")}`}
        </div>
      </TableCell>
      <TableCell>
        <div
          role="meter"
          aria-label={t("customerPortal.analytics.customerSpendShare")}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={sharePercentage}
          aria-valuetext={shareLabel}
          title={shareLabel}
          className="ml-auto h-2.5 w-24 overflow-hidden rounded-full bg-muted"
        >
          <div
            className="h-full rounded-full bg-primary"
            style={{ width: `${sharePercentage}%`, opacity: 0.3 + sharePercentage * 0.007 }}
          />
        </div>
      </TableCell>
      <TableCell className="text-right tabular-nums">{formatCompact(tokens)}</TableCell>
      <TableCell className="pr-0 text-right">
        <span
          className={cn(
            "inline-flex rounded-full bg-muted px-2 py-0.5 text-xs",
            atLimit && "bg-destructive/10 text-destructive",
          )}
        >
          {t(status)}
        </span>
      </TableCell>
    </>
  );
}
