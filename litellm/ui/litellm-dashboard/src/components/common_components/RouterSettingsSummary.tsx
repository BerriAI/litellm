import { useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";
import i18n from "@/i18n/i18n";
import { hasRouterSettings } from "./routerSettingsPayload";
import { routerWeightsFrom, savedRouterWeights } from "./routerWeights";

interface RouterSettingsSummaryProps {
  routerSettings: Record<string, unknown> | null | undefined;
  emptyText?: string;
}

const fallbackEntries = (fallbacks: unknown): Array<[string, string[]]> => {
  if (!Array.isArray(fallbacks)) return [];
  return fallbacks.flatMap((entry) =>
    entry && typeof entry === "object" ? (Object.entries(entry) as Array<[string, string[]]>) : [],
  );
};

export function RouterWeightsSummary({ weights }: { weights: unknown }) {
  const { t } = useTranslation();
  const saved = savedRouterWeights(routerWeightsFrom(weights));
  if (!saved) return null;
  return (
    <div>
      <div>{t("routerWeights.summary")}</div>
      <div className="mt-1 space-y-1">
        {Object.entries(saved).map(([group, groupWeights]) => (
          <div key={group} className="text-xs text-muted-foreground">
            <span className="font-medium">{group}</span>
            <span className="mx-1 text-muted-foreground">-&gt;</span>
            {Object.entries(groupWeights)
              .map(([deploymentId, weight]) => `${deploymentId}: ${weight}`)
              .join(", ")}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function RouterSettingsSummary({
  routerSettings,
  emptyText = i18n.t("routerSettingsSummary.empty"),
}: RouterSettingsSummaryProps) {
  const { t } = useTranslation();
  if (!hasRouterSettings(routerSettings)) {
    return <div className="text-muted-foreground">{emptyText}</div>;
  }

  const settings = routerSettings as Record<string, unknown>;
  const fallbacks = fallbackEntries(settings.fallbacks);

  return (
    <div className="space-y-1 text-sm">
      {settings.routing_strategy != null && (
        <div>
          {t("routerSettingsSummary.routingStrategy")}{" "}
          <Badge variant="secondary">{String(settings.routing_strategy)}</Badge>
        </div>
      )}
      {settings.num_retries != null && (
        <div>{t("routerSettingsSummary.numRetries", { value: String(settings.num_retries) })}</div>
      )}
      {settings.allowed_fails != null && (
        <div>{t("routerSettingsSummary.allowedFailures", { value: String(settings.allowed_fails) })}</div>
      )}
      {settings.cooldown_time != null && (
        <div>{t("routerSettingsSummary.cooldownTime", { value: String(settings.cooldown_time) })}</div>
      )}
      {settings.timeout != null && <div>{t("routerSettingsSummary.timeout", { value: String(settings.timeout) })}</div>}
      {settings.retry_after != null && (
        <div>{t("routerSettingsSummary.retryAfter", { value: String(settings.retry_after) })}</div>
      )}
      {Boolean(settings.enable_tag_filtering) && <div>{t("routerSettingsSummary.tagFilteringEnabled")}</div>}
      {fallbacks.length > 0 && (
        <div>
          <div>{t("routerSettingsSummary.fallbacks")}</div>
          <div className="mt-1 space-y-1">
            {fallbacks.map(([model, targets]) => (
              <div key={model} className="text-xs text-muted-foreground">
                <span className="font-medium">{model}</span>
                <span className="mx-1 text-muted-foreground">-&gt;</span>
                {Array.isArray(targets) ? targets.join(", ") : String(targets)}
              </div>
            ))}
          </div>
        </div>
      )}
      <RouterWeightsSummary weights={settings.weights} />
    </div>
  );
}
