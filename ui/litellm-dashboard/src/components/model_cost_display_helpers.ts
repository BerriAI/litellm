export type ModelCostDisplayState = "available" | "disabled" | "gated";

export interface FormatModelCostEstimateOptions {
  readonly status?: ModelCostDisplayState;
  readonly precision?: number;
}

const COST_PENDING_LABEL = "Cost pending";
const ESTIMATE_UNAVAILABLE_LABEL = "Estimate unavailable";

export const formatModelCostEstimate = (
  costPerToken: number | null | undefined,
  options: FormatModelCostEstimateOptions = {},
): string => {
  if (options.status === "disabled" || options.status === "gated") {
    return COST_PENDING_LABEL;
  }

  if (costPerToken == null) {
    return ESTIMATE_UNAVAILABLE_LABEL;
  }

  return (costPerToken * 1_000_000).toFixed(options.precision ?? 4);
};
