import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import i18n from "@/i18n/i18n";
import { classificationFrequency, type ComplexityRouterConfigValue } from "./ComplexityRouterConfig";

const blockedReason = (value: ComplexityRouterConfigValue, t: TFunction = i18n.t.bind(i18n)): string | null => {
  if (value.custom_tier_set) return t("addModel.cacheAwareRoutingConfig.blockedByCustomTiers");
  if (value.adaptive) return t("addModel.cacheAwareRoutingConfig.blockedByAdaptive");
  if (classificationFrequency(value) !== "every_request") {
    return t("addModel.cacheAwareRoutingConfig.blockedByFrequency");
  }
  if (Object.values(value.tiers).some((models) => models.length > 1)) {
    return t("addModel.cacheAwareRoutingConfig.blockedByMultipleModels");
  }
  if (
    Object.values(value.tier_model_params ?? {}).some((models) =>
      Object.values(models).some((params) => Object.keys(params).length > 0),
    )
  ) {
    return t("addModel.cacheAwareRoutingConfig.blockedByParamOverrides");
  }
  return null;
};

const optionalInteger = (raw: string, minimum: number): number | undefined => {
  if (raw.trim() === "") return undefined;
  const parsed = Number(raw);
  return Number.isFinite(parsed) ? Math.max(minimum, Math.trunc(parsed)) : undefined;
};

const CacheAwareRoutingConfig = ({
  value,
  onChange,
}: {
  value: ComplexityRouterConfigValue;
  onChange: (value: ComplexityRouterConfigValue) => void;
}) => {
  const { t } = useTranslation();
  const enabled = value.cache_aware_routing ?? false;
  const reason = blockedReason(value, t);
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Switch
          checked={enabled}
          disabled={!enabled && reason !== null}
          onCheckedChange={(next) => onChange({ ...value, cache_aware_routing: next })}
          aria-label={t("addModel.cacheAwareRoutingConfig.toggle")}
        />
        <span className="text-sm font-medium">{t("addModel.cacheAwareRoutingConfig.considerSavings")}</span>
      </div>
      <p className="text-xs text-muted-foreground">{t("addModel.cacheAwareRoutingConfig.description")}</p>
      {reason && (
        <p className="text-xs text-muted-foreground" role="status">
          {enabled && t("addModel.cacheAwareRoutingConfig.currentlySkipped")}
          {reason}
        </p>
      )}
      {enabled && (
        <div className="grid gap-4 sm:grid-cols-2">
          <div>
            <label className="block text-sm font-medium mb-1" htmlFor="cache-aware-output-tokens">
              {t("addModel.cacheAwareRoutingConfig.outputTokens")}
            </label>
            <Input
              id="cache-aware-output-tokens"
              type="number"
              min={0}
              step={1}
              placeholder="1024"
              value={value.cache_aware_routing_output_tokens ?? ""}
              onChange={(event) =>
                onChange({
                  ...value,
                  cache_aware_routing_output_tokens: optionalInteger(event.target.value, 0),
                })
              }
              aria-describedby="cache-aware-output-help"
            />
            <p id="cache-aware-output-help" className="mt-1 text-xs text-muted-foreground">
              {t("addModel.cacheAwareRoutingConfig.outputTokensHelp")}
            </p>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1" htmlFor="cache-aware-timeout">
              {t("addModel.cacheAwareRoutingConfig.timeout")}
            </label>
            <Input
              id="cache-aware-timeout"
              type="number"
              min={1}
              step={1}
              placeholder="2000"
              value={value.cache_aware_routing_timeout_ms ?? ""}
              onChange={(event) =>
                onChange({
                  ...value,
                  cache_aware_routing_timeout_ms: optionalInteger(event.target.value, 1),
                })
              }
              aria-describedby="cache-aware-timeout-help"
            />
            <p id="cache-aware-timeout-help" className="mt-1 text-xs text-muted-foreground">
              {t("addModel.cacheAwareRoutingConfig.timeoutHelp")}
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default CacheAwareRoutingConfig;
