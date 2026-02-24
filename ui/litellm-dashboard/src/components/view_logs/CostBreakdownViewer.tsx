import React from "react";
import { Collapse } from "antd";
import { formatNumberWithCommas } from "@/utils/dataUtils";

export interface CostBreakdown {
  input_cost?: number;
  output_cost?: number;
  total_cost?: number;
  tool_usage_cost?: number;
  additional_costs?: Record<string, number>;
  original_cost?: number;
  discount_percent?: number;
  discount_amount?: number;
  margin_percent?: number;
  margin_fixed_amount?: number;
  margin_total_amount?: number;
}

interface CostBreakdownViewerProps {
  costBreakdown: CostBreakdown | null | undefined;
  totalSpend: number;
  promptTokens?: number;
  completionTokens?: number;
  cacheHit?: string;
}

const formatCost = (cost: number | undefined): string => {
  if (cost === undefined || cost === null) return "-";
  return `$${formatNumberWithCommas(cost, 8)}`;
};

const formatPercent = (percent: number | undefined): string => {
  if (percent === undefined || percent === null) return "-";
  return `${(percent * 100).toFixed(2)}%`;
};

export const CostBreakdownViewer: React.FC<CostBreakdownViewerProps> = ({
  costBreakdown,
  totalSpend,
  promptTokens,
  completionTokens,
  cacheHit,
}) => {
  const isCached = cacheHit?.toLowerCase() === "true";
  const hasTokenCounts = promptTokens !== undefined || completionTokens !== undefined;

  const hasCostBreakdown = costBreakdown?.input_cost !== undefined || costBreakdown?.output_cost !== undefined;
  const hasMeaningfulData =
    hasCostBreakdown ||
    hasTokenCounts ||
    (costBreakdown &&
      ((costBreakdown.discount_percent !== undefined && costBreakdown.discount_percent !== 0) ||
        (costBreakdown.discount_amount !== undefined && costBreakdown.discount_amount !== 0) ||
        (costBreakdown.margin_percent !== undefined && costBreakdown.margin_percent !== 0) ||
        (costBreakdown.margin_fixed_amount !== undefined && costBreakdown.margin_fixed_amount !== 0) ||
        (costBreakdown.margin_total_amount !== undefined && costBreakdown.margin_total_amount !== 0)));

  if (!hasMeaningfulData) {
    return null;
  }

  const hasDiscount =
    costBreakdown &&
    ((costBreakdown.discount_percent !== undefined && costBreakdown.discount_percent !== 0) ||
      (costBreakdown.discount_amount !== undefined && costBreakdown.discount_amount !== 0));

  const hasMargin =
    costBreakdown &&
    ((costBreakdown.margin_percent !== undefined && costBreakdown.margin_percent !== 0) ||
      (costBreakdown.margin_fixed_amount !== undefined && costBreakdown.margin_fixed_amount !== 0) ||
      (costBreakdown.margin_total_amount !== undefined && costBreakdown.margin_total_amount !== 0));

  // When cached, show $0 (authoritative total) instead of pre-cache costs from cost_breakdown
  const inputCost = isCached ? 0 : costBreakdown?.input_cost;
  const outputCost = isCached ? 0 : costBreakdown?.output_cost;
  const originalCost = isCached ? 0 : costBreakdown?.original_cost;
  const totalCost = isCached ? 0 : (costBreakdown?.total_cost ?? totalSpend);

  return (
    <div className="bg-white rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
      <Collapse
        expandIconPosition="start"
        items={[
          {
            key: "1",
            label: (
              <div className="flex items-center justify-between w-full">
                <h3 className="text-lg font-medium text-gray-900">Cost Breakdown</h3>
                <div className="flex items-center space-x-2 mr-4">
                  <span className="text-sm text-gray-500">Total:</span>
                  <span className="text-sm font-semibold text-gray-900">
                    {formatCost(totalSpend)}
                    {isCached && " (Cached)"}
                  </span>
                </div>
              </div>
            ),
            children: (
              <div className="p-6 space-y-4">
            {/* Step 1: Base Token Costs */}
            <div className="space-y-2 max-w-2xl">
              <div className="flex text-sm">
                <span className="text-gray-600 font-medium w-1/3">Input Cost:</span>
                <span className="text-gray-900">
                  {formatCost(inputCost)}
                  {promptTokens !== undefined && (
                    <span className="text-gray-500 font-normal ml-1">
                      ({promptTokens.toLocaleString()} prompt tokens)
                    </span>
                  )}
                </span>
              </div>
              <div className="flex text-sm">
                <span className="text-gray-600 font-medium w-1/3">Output Cost:</span>
                <span className="text-gray-900">
                  {formatCost(outputCost)}
                  {completionTokens !== undefined && (
                    <span className="text-gray-500 font-normal ml-1">
                      ({completionTokens.toLocaleString()} completion tokens)
                    </span>
                  )}
                </span>
              </div>
              {costBreakdown?.tool_usage_cost !== undefined && costBreakdown.tool_usage_cost > 0 && (
                <div className="flex text-sm">
                  <span className="text-gray-600 font-medium w-1/3">Tool Usage Cost:</span>
                  <span className="text-gray-900">{formatCost(costBreakdown.tool_usage_cost)}</span>
                </div>
              )}
              {/* Additional Costs (free-form) */}
              {costBreakdown?.additional_costs && Object.keys(costBreakdown.additional_costs).length > 0 && (
                <>
                  {Object.entries(costBreakdown.additional_costs).map(([key, value]) => (
                    <div key={key} className="flex text-sm">
                      <span className="text-gray-600 font-medium w-1/3">{key}:</span>
                      <span className="text-gray-900">{formatCost(value)}</span>
                    </div>
                  ))}
                </>
              )}
            </div>

            {/* Subtotal / Original Cost - hide when cached since it would be $0 */}
            {!isCached && (
              <div className="pt-2 border-t border-gray-100 max-w-2xl">
                <div className="flex text-sm font-semibold">
                  <span className="text-gray-900 w-1/3">Original LLM Cost:</span>
                  <span className="text-gray-900">{formatCost(originalCost)}</span>
                </div>
              </div>
            )}

            {/* Step 2: Adjustments (Discount & Margin) */}
            {(hasDiscount || hasMargin) && (
              <div className="pt-2 space-y-2 max-w-2xl">
                {/* Discounts */}
                {hasDiscount && (
                  <div className="space-y-2">
                    {costBreakdown.discount_percent !== undefined && costBreakdown.discount_percent !== 0 && (
                      <div className="flex text-sm text-gray-600">
                        <span className="font-medium w-1/3">Discount ({formatPercent(costBreakdown.discount_percent)}):</span>
                        <span className="text-gray-900">-{formatCost(costBreakdown.discount_amount)}</span>
                      </div>
                    )}
                    {costBreakdown.discount_amount !== undefined && costBreakdown.discount_percent === undefined && (
                      <div className="flex text-sm text-gray-600">
                        <span className="font-medium w-1/3">Discount Amount:</span>
                        <span className="text-gray-900">-{formatCost(costBreakdown.discount_amount)}</span>
                      </div>
                    )}
                  </div>
                )}

                {/* Margins */}
                {hasMargin && (
                  <div className="space-y-2">
                    {costBreakdown.margin_percent !== undefined && costBreakdown.margin_percent !== 0 && (
                      <div className="flex text-sm text-gray-600">
                        <span className="font-medium w-1/3">Margin ({formatPercent(costBreakdown.margin_percent)}):</span>
                        <span className="text-gray-900">+{formatCost((costBreakdown.margin_total_amount || 0) - (costBreakdown.margin_fixed_amount || 0))}</span>
                      </div>
                    )}
                    {costBreakdown.margin_fixed_amount !== undefined && costBreakdown.margin_fixed_amount !== 0 && (
                      <div className="flex text-sm text-gray-600">
                        <span className="font-medium w-1/3">Margin:</span>
                        <span className="text-gray-900">+{formatCost(costBreakdown.margin_fixed_amount)}</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Final Summary */}
            <div className="mt-4 pt-4 border-t border-gray-200 max-w-2xl">
              <div className="flex items-center">
                <span className="font-bold text-sm text-gray-900 w-1/3">Final Calculated Cost:</span>
                <span className="text-sm font-bold text-gray-900">
                  {formatCost(totalCost)}
                  {isCached && " (Cached)"}
                </span>
              </div>
            </div>
          </div>
            ),
          },
        ]}
      />
    </div>
  );
};

export default CostBreakdownViewer;
