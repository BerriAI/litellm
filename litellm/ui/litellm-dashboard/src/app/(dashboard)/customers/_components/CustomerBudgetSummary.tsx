import { useTranslation } from "react-i18next";
import type { CustomerUsageEntity } from "./customerUsageApi";
import CustomerBudgetSpend, { CustomerBudgetWindow } from "./CustomerBudgetSpend";
import { CUSTOMER_BUDGET_LABELS, customerBudgetPeriod, customerBudgetUsage } from "./customerBudgetUsage";

export default function CustomerBudgetSummary({
  entity,
  settingsHref,
}: {
  entity: CustomerUsageEntity;
  settingsHref: string;
}) {
  const { t } = useTranslation();
  const budgets = customerBudgetUsage(entity);
  return (
    <section className="mt-2 min-w-0 space-y-2 text-sm" aria-label={t("customerPortal.analytics.budgetSummary")}>
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-xs">
        <span className="text-muted-foreground">{t("customerPortal.analytics.budgetSummary")}</span>
        <a href={settingsHref} className="font-medium text-primary hover:underline">
          {t(budgets.length > 0 ? "customerPortal.analytics.editBudget" : "customerPortal.analytics.setBudget")}
        </a>
      </div>
      {budgets.map((budget) => {
        const period = customerBudgetPeriod(budget.budget_duration);
        return (
          <div key={`${budget.source}-${budget.budget_duration}`} className="space-y-1">
            <p className="text-[11px] text-muted-foreground">
              {t(CUSTOMER_BUDGET_LABELS[period])}
              {period === "other" && budget.budget_duration && ` (${budget.budget_duration})`}
            </p>
            <CustomerBudgetWindow
              budget={budget}
              showReset={budgets.filter((item) => customerBudgetPeriod(item.budget_duration) === period).length > 1}
            />
          </div>
        );
      })}
      {budgets.length === 0 && (
        <CustomerBudgetSpend spend={entity.spend} limit={null} label={t("customerPortal.analytics.budgetSummary")} />
      )}
      <p className="text-[11px] text-muted-foreground">{t("customerPortal.analytics.budgetUsageHint")}</p>
    </section>
  );
}
