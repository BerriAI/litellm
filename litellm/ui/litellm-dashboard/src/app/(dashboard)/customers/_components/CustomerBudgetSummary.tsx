import { useTranslation } from "react-i18next";
import { useState } from "react";
import type { CustomerUsageEntity } from "./customerUsageApi";
import { CustomerBudgetWindow } from "./CustomerBudgetSpend";
import { CUSTOMER_BUDGET_LABELS, customerBudgetPeriod, customerBudgetUsage } from "./customerBudgetUsage";
import { CustomerRateLimitSummary } from "./CustomerRateLimits";
import CustomerEntityBudgetDialog from "./CustomerEntityBudgetDialog";
import type { CustomerHierarchyScope } from "./customerHierarchyApi";

export default function CustomerBudgetSummary({
  entity,
  settingsHref,
  scope,
}: {
  entity: CustomerUsageEntity;
  settingsHref: string;
  scope: CustomerHierarchyScope;
}) {
  const { t } = useTranslation();
  const budgets = customerBudgetUsage(entity);
  const organizationId = !scope.team && !scope.member ? scope.org : undefined;
  const [editing, setEditing] = useState(false);
  const editLabel = t(
    budgets.length > 0 || entity.rateLimits
      ? "customerPortal.analytics.editBudget"
      : "customerPortal.analytics.setBudget",
  );
  const showBudgetDialog = editing && Boolean(organizationId || scope.member);
  return (
    <section className="mt-2 min-w-0 space-y-2 text-sm" aria-label={t("customerPortal.analytics.budgetSummary")}>
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-xs">
        <span className="text-muted-foreground">{t("customerPortal.analytics.budgetSummary")}</span>
        {organizationId || scope.member ? (
          <button type="button" onClick={() => setEditing(true)} className="font-medium text-primary hover:underline">
            {editLabel}
          </button>
        ) : (
          <a href={settingsHref} className="font-medium text-primary hover:underline">
            {editLabel}
          </a>
        )}
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
        <p className="text-muted-foreground">
          {t(entity.rateLimits ? "customerPortal.rateLimits.noSpendLimit" : "customerPortal.analytics.noBudget")}
        </p>
      )}
      <CustomerRateLimitSummary limits={entity.rateLimits} />
      {showBudgetDialog && (
        <CustomerEntityBudgetDialog
          key={scope.member}
          memberId={scope.member ?? undefined}
          organizationId={scope.org}
          onClose={() => setEditing(false)}
        />
      )}
    </section>
  );
}
