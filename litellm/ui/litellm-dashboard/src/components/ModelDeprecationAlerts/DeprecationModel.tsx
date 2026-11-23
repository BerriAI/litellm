import { useTranslation } from "react-i18next";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import type { DeprecationAlert } from "./deprecationAlerts";
import { migrationHref } from "./migrationRoute";

export function deprecationProvider(alert: DeprecationAlert): string {
  return alert.model.litellm_provider || alert.usageModel.split("/")[0];
}

export function DeprecationModel({ alert }: { alert: DeprecationAlert }) {
  const provider = deprecationProvider(alert);
  return (
    <a href={migrationHref(alert.usageModel)} className="inline-flex items-center gap-2 font-medium hover:underline">
      <ProviderLogo provider={provider} className="size-4 shrink-0" />
      {alert.usageModel}
    </a>
  );
}

export function DeprecationTiming({ alert }: { alert: DeprecationAlert }) {
  const { t } = useTranslation();
  const days = alert.model.days_until_deprecation;
  let label = t("alerts.timing.deprecatesIn", { count: days });
  if (days === 0) label = t("alerts.timing.today");
  if (days < 0) label = t("alerts.timing.deprecatedAgo", { count: Math.abs(days) });
  return (
    <span
      className={days < 0 ? "text-destructive" : "text-amber-700 dark:text-amber-400"}
      title={alert.model.deprecation_date}
    >
      {label}
    </span>
  );
}
