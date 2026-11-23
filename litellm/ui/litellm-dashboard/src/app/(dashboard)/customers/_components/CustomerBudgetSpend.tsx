import { useTranslation } from "react-i18next";
import { BudgetUsageProgress } from "@/components/shared/BudgetUsageProgress";
import { formatUsd } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { CUSTOMER_BUDGET_LABELS, customerBudgetPeriod, type CustomerBudgetUsage } from "./customerBudgetUsage";

function budgetPercentage(spend: number, limit: number | null): number {
  if (limit === null) return 0;
  if (limit === 0) return 100;
  return Math.max(0, (spend / limit) * 100);
}

export function CustomerBudgetWindow({
  budget,
  showReset = false,
  stacked = false,
}: {
  budget: CustomerBudgetUsage;
  showReset?: boolean;
  stacked?: boolean;
}) {
  const { t, i18n } = useTranslation();
  const period = customerBudgetPeriod(budget.budget_duration);
  const label = t(CUSTOMER_BUDGET_LABELS[period]);
  const reset = budget.reset_at
    ? t("customerPortal.analytics.budgetResets", {
        date: new Date(budget.reset_at).toLocaleString(i18n.resolvedLanguage, { timeZone: "UTC" }),
      })
    : null;
  return (
    <div title={[budget.budget_duration, reset].filter(Boolean).join(" · ")}>
      <CustomerBudgetSpend
        spend={budget.spend}
        limit={budget.max_budget}
        label={label}
        compact={period === "yearly"}
        stacked={stacked}
      />
      {showReset && reset && <p className="mt-1 text-[11px] text-muted-foreground">{reset}</p>}
    </div>
  );
}

export default function CustomerBudgetSpend({
  spend,
  limit,
  label,
  compact = false,
  stacked = false,
  showLimit = true,
  limitKnown = true,
  noLimitLabel,
}: {
  spend: number | null;
  limit: number | null;
  label: string;
  compact?: boolean;
  stacked?: boolean;
  showLimit?: boolean;
  limitKnown?: boolean;
  noLimitLabel?: string;
}) {
  const { t } = useTranslation();
  const spendValue = spend === null ? t("customerPortal.analytics.unknownSpend") : formatUsd(spend);
  const unlimitedLabel = limitKnown
    ? noLimitLabel ?? t("customerPortal.analytics.noLimit")
    : t("customerPortal.analytics.unknownSpend");
  if (limit === null && !showLimit) return <span className="tabular-nums">{spendValue}</span>;
  if (limit === null || stacked) {
    return (
      <div className="space-y-0.5 whitespace-nowrap tabular-nums">
        <div>{spendValue}</div>
        <div className="text-xs text-muted-foreground">
          {limit === null ? unlimitedLabel : t("customerPortal.analytics.limitAmount", { amount: formatUsd(limit) })}
        </div>
      </div>
    );
  }
  const value = `${spendValue} / ${formatUsd(limit)}`;
  const percentage = budgetPercentage(spend ?? 0, limit);
  const usage =
    limit === 0
      ? t("customerPortal.list.zeroBudgetPercentage")
      : t("customerPortal.list.budgetUsed", {
          percentage: t("format.number", { val: percentage / 100, style: "percent", maximumFractionDigits: 0 }),
        });
  const balance =
    spend === null
      ? null
      : t(spend > limit ? "customerPortal.analytics.budgetOver" : "customerPortal.analytics.budgetRemaining", {
          amount: formatUsd(Math.abs(limit - spend)),
        });
  if (compact) return <span className="whitespace-nowrap tabular-nums">{value}</span>;
  return (
    <div className="space-y-2 tabular-nums">
      <div className="whitespace-nowrap font-medium">{value}</div>
      {spend !== null && (
        <>
          <BudgetUsageProgress percentage={percentage} label={label} valueText={`${value}, ${balance}`} />
          <div className="flex items-start justify-between gap-3 text-xs text-muted-foreground">
            <span>{usage}</span>
            <span className="whitespace-nowrap text-right">{balance}</span>
          </div>
        </>
      )}
    </div>
  );
}
