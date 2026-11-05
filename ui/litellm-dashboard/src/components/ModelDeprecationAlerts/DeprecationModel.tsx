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

export function DeprecationTiming({
  alert,
  format = "sentence",
}: {
  alert: DeprecationAlert;
  format?: "sentence" | "duration";
}) {
  const days = alert.model.days_until_deprecation;
  if (format === "duration") {
    if (days === 0) return <span title={alert.model.deprecation_date}>Today</span>;
    return (
      <span title={alert.model.deprecation_date}>
        {Math.abs(days)}{" "}
        <span className="text-sm font-normal text-muted-foreground">
          {Math.abs(days) === 1 ? "day" : "days"}
          {days < 0 ? " ago" : ""}
        </span>
      </span>
    );
  }
  let label = `Deprecates in ${days} ${days === 1 ? "day" : "days"}`;
  if (days === 0) label = "Deprecates today";
  if (days < 0) label = `Deprecated ${Math.abs(days)} ${days === -1 ? "day" : "days"} ago`;
  return (
    <span
      className={days < 0 ? "text-destructive" : "text-amber-700 dark:text-amber-400"}
      title={alert.model.deprecation_date}
    >
      {label}
    </span>
  );
}
