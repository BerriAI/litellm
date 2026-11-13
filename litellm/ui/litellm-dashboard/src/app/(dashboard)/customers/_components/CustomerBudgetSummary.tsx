import { useTranslation } from "react-i18next";
import { formatUsd } from "@/app/(dashboard)/usage/_components/components/overview/overviewData";
import { cn } from "@/lib/cva.config";
import type { CustomerUsageEntity } from "./customerUsageApi";

function budgetPercentage(spend: number, limit: number | null): number {
  if (limit === null) return 0;
  if (limit === 0) return 100;
  return Math.max(0, Math.min(100, (spend / limit) * 100));
}

export default function CustomerBudgetSummary({
  entity,
  settingsHref,
}: {
  entity: CustomerUsageEntity;
  settingsHref: string;
}) {
  const { t, i18n } = useTranslation();
  const limit = entity.budgetLimit;
  const atLimit = limit !== null && entity.spend >= limit;
  const percentage = budgetPercentage(entity.spend, limit);
  const cap = limit === null ? t("customerPortal.analytics.noCap") : formatUsd(limit);
  const value = `${formatUsd(entity.spend)} / ${cap}`;
  return (
    <section className="mt-3 min-w-0 space-y-1.5" aria-label={t("customerPortal.analytics.budgetSummary")}>
      <div className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 text-xs">
        <span className="text-muted-foreground">{t("customerPortal.analytics.budgetSummary")}</span>
        <a href={settingsHref} className="font-medium text-primary hover:underline">
          {t(limit === null ? "customerPortal.analytics.setBudget" : "customerPortal.analytics.editBudget")}
        </a>
      </div>
      <p className={cn("text-sm font-medium tabular-nums", atLimit && "text-destructive")}>{value}</p>
      {limit !== null && (
        <div
          role="progressbar"
          aria-label={t("customerPortal.analytics.budgetSummary")}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={percentage}
          aria-valuetext={value}
          className="h-1 overflow-hidden rounded-full bg-muted"
        >
          <div className={cn("h-full bg-primary", atLimit && "bg-destructive")} style={{ width: `${percentage}%` }} />
        </div>
      )}
      <p className="text-[11px] text-muted-foreground">
        {t("customerPortal.analytics.budgetPeriod")}
        {entity.budgetResetAt &&
          ` · ${t("customerPortal.analytics.budgetResets", {
            date: new Date(entity.budgetResetAt).toLocaleDateString(i18n.resolvedLanguage, { timeZone: "UTC" }),
          })}`}
      </p>
    </section>
  );
}
