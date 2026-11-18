import { useTranslation } from "react-i18next";
import {
  CUSTOMER_BUDGET_LABELS,
  customerBudgetPeriod,
  type CustomerBudgetUsage,
} from "../../customers/_components/customerBudgetUsage";
import { CustomerBudgetWindow } from "../../customers/_components/CustomerBudgetSpend";
import { appSpend } from "./appApi";
import type { Budget } from "./appForms";

export default function AppBudgetLimits({
  budget,
  usage,
}: {
  budget?: Budget | null;
  usage?: readonly CustomerBudgetUsage[] | null;
}) {
  const { t } = useTranslation();
  const scalar =
    budget?.max_budget == null
      ? []
      : [{ max_budget: budget.max_budget, budget_duration: budget.budget_duration ?? null }];
  const limits = [...scalar, ...(budget?.budget_limits ?? [])];
  const recordedUsage = usage ?? [];
  const hasUsage = recordedUsage.length > 0;
  const periodLabel = (duration: string | null) => {
    const period = customerBudgetPeriod(duration);
    if (period === "monthly") return t("apps.month");
    if (period === "yearly") return t("apps.year");
    return duration;
  };
  return (
    <div className="space-y-1 tabular-nums">
      {hasUsage &&
        recordedUsage.map((window, index) => (
          <div key={`${window.source}-${window.budget_duration}-${index}`} className="space-y-1 py-1">
            <p className="text-xs text-muted-foreground">
              {t(CUSTOMER_BUDGET_LABELS[customerBudgetPeriod(window.budget_duration)])}
            </p>
            <CustomerBudgetWindow budget={window} showReset />
          </div>
        ))}
      {!hasUsage && limits.length === 0 && <span>{t("apps.unlimited")}</span>}
      {!hasUsage &&
        limits.map((limit, index) => (
          <div key={`${limit.budget_duration}-${index}`}>
            <span>{appSpend(limit.max_budget)}</span>
            {limit.budget_duration && (
              <span className="text-xs text-muted-foreground"> / {periodLabel(limit.budget_duration)}</span>
            )}
          </div>
        ))}
      {budget?.rpm_limit != null && (
        <span className="block text-xs text-muted-foreground">{t("apps.rpmLimit", { count: budget.rpm_limit })}</span>
      )}
    </div>
  );
}
