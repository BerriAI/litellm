import React, { useCallback } from "react";
import PricingForm from "./pricing_form";
import CostResults from "./cost_results";
import { useCostEstimate } from "./use_cost_estimate";
import { PricingCalculatorProps, PricingFormValues } from "./types";

const PricingCalculator: React.FC<PricingCalculatorProps> = ({
  accessToken,
  models,
}) => {
  const { loading, result, debouncedFetch } = useCostEstimate(accessToken);

  const handleValuesChange = useCallback(
    (_changedValues: Partial<PricingFormValues>, allValues: PricingFormValues) => {
      if (allValues.model) {
        debouncedFetch(allValues);
      }
    },
    [debouncedFetch]
  );

  return (
    <div className="space-y-6">
      <PricingForm models={models} onValuesChange={handleValuesChange} />
      <CostResults result={result} loading={loading} />
    </div>
  );
};

export default PricingCalculator;

