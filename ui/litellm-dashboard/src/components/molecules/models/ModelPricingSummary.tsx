import { useTranslation } from "react-i18next";

import { ModelData, ModelInfo } from "@/components/model_dashboard/types";
import { Badge } from "@/components/ui/badge";
import { formatPerMillionCost, formatPerSecondCost } from "@/utils/dataUtils";

type PricingFields = Pick<
  ModelData,
  "input_cost" | "output_cost" | "output_cost_per_second" | "output_cost_per_second_tiers"
> & { model_info?: Pick<ModelInfo, "pricing_overrides"> };

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
  const { input_cost: inputCost, output_cost: outputCost, output_cost_per_second: perSecond } = model;
  const hasPerSecond = perSecond != null;
  const showInput = inputCost != null && (!hasPerSecond || Number(inputCost) > 0);
  const showOutput = outputCost != null && (!hasPerSecond || Number(outputCost) > 0);

  if (!showInput && !showOutput && !hasPerSecond) {
    return <p className="mt-2 text-sm text-muted-foreground">-</p>;
  }

  return (
    <div className="mt-2">
      {showInput && (
        <p className="text-sm">{t("modelPricingSummary.inputPerMillion", { cost: formatPerMillionCost(inputCost) })}</p>
      )}
      {showOutput && (
        <p className="text-sm">
          {t("modelPricingSummary.outputPerMillion", { cost: formatPerMillionCost(outputCost) })}
        </p>
      )}
      {hasPerSecond && (
        <p className="text-sm">{t("modelPricingSummary.outputPerSecond", { cost: formatPerSecondCost(perSecond) })}</p>
      )}
      {(model.output_cost_per_second_tiers ?? []).map(({ resolution, cost }) => (
        <p key={resolution} className="text-sm">
          {t("modelPricingSummary.outputPerSecondTier", { resolution, cost: formatPerSecondCost(cost) })}
        </p>
      ))}
      <PricingSource overrides={model.model_info?.pricing_overrides} />
    </div>
  );
}
