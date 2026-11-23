import { useTranslation } from "react-i18next";
import { Info } from "lucide-react";
import { BudgetUsageProgress } from "@/components/shared/BudgetUsageProgress";
import { SimpleTooltip } from "@/components/ui/tooltip";
import { formatUsd } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { customerBudgetPeriod, type CustomerBudgetUsage } from "../../customers/_components/customerBudgetUsage";
import { appBudgetReached, appBudgetWindows, appSpend } from "./appBudgetPresentation";
import type { Budget } from "./appForms";
import { CustomerRateLimitSummary } from "../../customers/_components/CustomerRateLimits";

function BudgetDetails({
  windows,
  cumulativeSpend,
}: {
  windows: readonly CustomerBudgetUsage[];
  cumulativeSpend?: number;
}) {
  const { t, i18n } = useTranslation();
  const details = (
    <div className="space-y-2">
      {windows.map((window, index) => {
        const period = customerBudgetPeriod(window.budget_duration);
        const caption = period === "other" ? window.budget_duration ?? t("apps.noReset") : t(`apps.${period}`);
        return (
          <p key={`${window.source}-${window.budget_duration}-${index}`}>
            <span className="font-medium">{caption}</span>
            {": "}
            {window.spend === null
              ? formatUsd(window.max_budget)
              : t("apps.usedOf", { used: appSpend(window.spend), limit: formatUsd(window.max_budget) })}
            {window.reset_at && (
              <span className="block opacity-75">
                {t("apps.resetsOn", {
                  date: new Date(window.reset_at).toLocaleDateString(i18n.resolvedLanguage, {
                    month: "short",
                    day: "numeric",
                    timeZone: "UTC",
                  }),
                })}
              </span>
            )}
          </p>
        );
      })}
      {cumulativeSpend !== undefined && (
        <p className="opacity-75">{t("apps.lifetimeTotal", { amount: appSpend(cumulativeSpend) })}</p>
      )}
    </div>
  );
  return (
    <SimpleTooltip content={details}>
      <button
        type="button"
        aria-label={t("apps.budgetDetails")}
        className="rounded p-0.5 text-muted-foreground hover:text-foreground"
      >
        <Info className="size-3.5" />
      </button>
    </SimpleTooltip>
  );
}

export default function AppBudgetLimits({
  budget,
  usage,
  cumulativeSpend,
}: {
  budget?: Budget | null;
  usage?: readonly CustomerBudgetUsage[] | null;
  cumulativeSpend?: number;
}) {
  const { t } = useTranslation();
  const { primary, windows } = appBudgetWindows(budget, usage);
  if (!primary)
    return (
      <div className="space-y-1 text-sm">
        <span>{t("customerPortal.rateLimits.noSpendLimit")}</span>
        <CustomerRateLimitSummary limits={budget} />
        {cumulativeSpend !== undefined && (
          <p className="text-xs text-muted-foreground">
            <span>{appSpend(cumulativeSpend)}</span> {t("apps.lifetimeSpend")}
          </p>
        )}
      </div>
    );
  const preview = usage == null && cumulativeSpend === undefined;
  const spent = primary.spend;
  const atLimit = appBudgetReached(primary);
  const percentage =
    primary.max_budget === 0 ? 100 : Math.min(100, Math.max(0, ((spent ?? 0) / primary.max_budget) * 100));
  const remaining = spent === null ? null : Math.max(0, primary.max_budget - spent);
  const label =
    spent === null
      ? formatUsd(primary.max_budget)
      : t("apps.usedOf", { used: appSpend(spent), limit: formatUsd(primary.max_budget) });
  const precise = spent !== null && spent > 0 && spent < 0.01 ? `$${spent}` : null;
  const period = customerBudgetPeriod(primary.budget_duration);
  const periodLabel = period === "other" ? primary.budget_duration : t(`apps.${period}`);
  let caption = t("apps.periodUsageUnavailable");
  if (preview) caption = periodLabel ?? t("apps.noReset");
  else if (atLimit) caption = t("apps.exhausted");
  else if (remaining !== null) caption = t("apps.remaining", { amount: appSpend(remaining) });
  return (
    <div className="min-w-56 max-w-sm space-y-1.5 tabular-nums">
      <div className="flex items-center gap-1.5">
        {precise ? (
          <SimpleTooltip content={t("apps.preciseSpend", { amount: precise })}>
            <span tabIndex={0} className="text-sm font-medium">
              {label}
            </span>
          </SimpleTooltip>
        ) : (
          <span className="text-sm font-medium">{label}</span>
        )}
        <BudgetDetails windows={windows} cumulativeSpend={cumulativeSpend} />
      </div>
      {spent !== null && <BudgetUsageProgress percentage={percentage} label={t("apps.budgetUsed")} valueText={label} />}
      <p className={atLimit ? "text-xs text-destructive" : "text-xs text-muted-foreground"}>{caption}</p>
      <CustomerRateLimitSummary limits={budget} />
    </div>
  );
}
