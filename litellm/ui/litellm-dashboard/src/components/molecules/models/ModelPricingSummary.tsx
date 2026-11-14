import { useTranslation } from "react-i18next";

import { LiteLLMParams, ModelData, ModelInfo } from "@/components/model_dashboard/types";
import { Badge } from "@/components/ui/badge";
import { formatPerSecondCost } from "@/utils/dataUtils";
import { formatAmount, storedCurrencySetting } from "@/utils/pricingCurrency";

type PricingFields = Pick<
  ModelData,
  "input_cost" | "output_cost" | "output_cost_per_second" | "output_cost_per_second_tiers"
> & {
  model_info?: Pick<ModelInfo, "pricing_overrides" | "input_cost_per_token" | "output_cost_per_token">;
  litellm_params?: LiteLLMParams;
};

const perMillion = (rate: unknown): string | null =>
  typeof rate === "number" && Number.isFinite(rate) ? formatAmount(rate * 1_000_000) : null;

function SourceCurrency({ model }: { model: PricingFields }) {
  const { t } = useTranslation();
  const setting = storedCurrencySetting(model.litellm_params, model.model_info);
  if (setting.currency === "USD") return null;
  const input = perMillion(model.litellm_params?.input_cost_per_token ?? model.model_info?.input_cost_per_token);
  const output = perMillion(model.litellm_params?.output_cost_per_token ?? model.model_info?.output_cost_per_token);
  return (
    <div className="mt-2 text-xs text-muted-foreground">
      <p>{t("pricingCurrency.configuredIn", { currency: setting.currency, rate: setting.rate })}</p>
      {input !== null && <p>{t("modelPricingSummary.configuredInput", { currency: setting.currency, cost: input })}</p>}
      {output !== null && (
        <p>{t("modelPricingSummary.configuredOutput", { currency: setting.currency, cost: output })}</p>
      )}
      <p>{t("pricingCurrency.spendInUsd")}</p>
    </div>
  );
}

function PricingSource({ overrides }: { overrides: string[] | undefined }) {
  const { t } = useTranslation();
  if (overrides === undefined) return null;
  if (overrides.length === 0) {
    return <p className="mt-2 text-xs text-muted-foreground">{t("modelPricingSummary.followsCostMap")}</p>;
  }
  return (
    <p className="mt-2 text-xs text-muted-foreground">
      <Badge variant="outline" className="mr-1">
        {t("modelPricingSummary.customPricing")}
      </Badge>
      {t("modelPricingSummary.overridesCostMap", { fields: overrides.join(", ") })}
    </p>
  );
}

export function ModelPricingSummary({ model }: { model: PricingFields }) {
  const { t } = useTranslation();
  const perSecond = model.output_cost_per_second;
  const hasPerSecond = perSecond != null;
  const showInput = model.input_cost != null && (!hasPerSecond || Number(model.input_cost) > 0);
  const showOutput = model.output_cost != null && (!hasPerSecond || Number(model.output_cost) > 0);

  if (!showInput && !showOutput && !hasPerSecond) {
    return <p className="mt-2 text-sm text-muted-foreground">-</p>;
  }

  return (
    <div className="mt-2">
      {showInput && <p className="text-sm">{t("modelPricingSummary.inputPerMillion", { cost: model.input_cost })}</p>}
      {showOutput && (
        <p className="text-sm">{t("modelPricingSummary.outputPerMillion", { cost: model.output_cost })}</p>
      )}
      {hasPerSecond && (
        <p className="text-sm">{t("modelPricingSummary.outputPerSecond", { cost: formatPerSecondCost(perSecond) })}</p>
      )}
      {(model.output_cost_per_second_tiers ?? []).map(({ resolution, cost }) => (
        <p key={resolution} className="text-sm">
          {t("modelPricingSummary.outputPerSecondTier", { resolution, cost: formatPerSecondCost(cost) })}
        </p>
      ))}
      <SourceCurrency model={model} />
      <PricingSource overrides={model.model_info?.pricing_overrides} />
    </div>
  );
}
