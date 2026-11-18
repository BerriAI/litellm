import { useTranslation } from "react-i18next";

const BREAKDOWN_DIGITS = { minimumFractionDigits: 2, maximumFractionDigits: 4 } as const;

export const customerSurchargeBreakdown = (spend: number, surcharge: number | null | undefined) => {
  const applied = Math.min(Math.max(surcharge ?? 0, 0), Math.max(spend, 0));
  return applied > 0 ? { baseCost: spend - applied, surcharge: applied } : null;
};

export default function CustomerSurchargeBreakdown({
  spend,
  surcharge,
}: {
  spend: number;
  surcharge: number | null | undefined;
}) {
  const { t } = useTranslation();
  const breakdown = customerSurchargeBreakdown(spend, surcharge);
  if (!breakdown) return null;
  return (
    <dl
      className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs leading-4 text-muted-foreground"
      aria-label={t("customerPortal.analytics.spendBreakdown")}
    >
      <div className="flex gap-1">
        <dt>{t("customerPortal.analytics.baseCost")}</dt>
        <dd className="tabular-nums text-foreground">
          {t("format.usd", { val: breakdown.baseCost, ...BREAKDOWN_DIGITS })}
        </dd>
      </div>
      <div className="flex gap-1">
        <dt>{t("customerPortal.analytics.surcharge")}</dt>
        <dd className="tabular-nums text-foreground">
          {t("format.usd", { val: breakdown.surcharge, ...BREAKDOWN_DIGITS })}
        </dd>
      </div>
    </dl>
  );
}
