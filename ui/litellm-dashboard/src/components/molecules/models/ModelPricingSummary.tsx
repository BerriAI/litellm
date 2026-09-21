import { ModelData, ModelInfo } from "@/components/model_dashboard/types";
import { Badge } from "@/components/ui/badge";
import { formatPerSecondCost } from "@/utils/dataUtils";

type PricingFields = Pick<
  ModelData,
  "input_cost" | "output_cost" | "output_cost_per_second" | "output_cost_per_second_tiers"
> & { model_info?: Pick<ModelInfo, "pricing_overrides"> };

function PricingSource({ overrides }: { overrides: string[] | undefined }) {
  if (overrides === undefined) return null;
  if (overrides.length === 0) {
    return <p className="mt-2 text-xs text-muted-foreground">Follows the model cost map</p>;
  }
  return (
    <p className="mt-2 text-xs text-muted-foreground">
      <Badge variant="outline" className="mr-1">
        Custom pricing
      </Badge>
      Overrides the model cost map for {overrides.join(", ")}
    </p>
  );
}

export function ModelPricingSummary({ model }: { model: PricingFields }) {
  const perSecond = model.output_cost_per_second;
  const hasPerSecond = perSecond != null;
  const showInput = model.input_cost != null && (!hasPerSecond || Number(model.input_cost) > 0);
  const showOutput = model.output_cost != null && (!hasPerSecond || Number(model.output_cost) > 0);

  if (!showInput && !showOutput && !hasPerSecond) {
    return <p className="mt-2 text-sm text-muted-foreground">-</p>;
  }

  return (
    <div className="mt-2">
      {showInput && <p className="text-sm">Input: ${model.input_cost}/1M tokens</p>}
      {showOutput && <p className="text-sm">Output: ${model.output_cost}/1M tokens</p>}
      {hasPerSecond && <p className="text-sm">Output: {formatPerSecondCost(perSecond)}</p>}
      {(model.output_cost_per_second_tiers ?? []).map(({ resolution, cost }) => (
        <p key={resolution} className="text-sm">
          Output ({resolution}): {formatPerSecondCost(cost)}
        </p>
      ))}
      <PricingSource overrides={model.model_info?.pricing_overrides} />
    </div>
  );
}
