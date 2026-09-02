import { formatPerSecondCost } from "@/app/(dashboard)/models-and-endpoints/utils/modelDataTransformer";
import { ModelData } from "@/components/model_dashboard/types";

type PricingFields = Pick<
  ModelData,
  "input_cost" | "output_cost" | "output_cost_per_second" | "output_cost_per_second_tiers"
>;

export function ModelPricingSummary({ model }: { model: PricingFields }) {
  const perSecond = model.output_cost_per_second;
  const hasPerSecond = perSecond != null;
  const showInput = !hasPerSecond || Number(model.input_cost) > 0;
  const showOutput = !hasPerSecond || Number(model.output_cost) > 0;

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
    </div>
  );
}
