import React from "react";
import { Accordion, AccordionHeader, AccordionBody } from "@tremor/react";
import { formatNumberWithCommas } from "@/utils/dataUtils";

export interface CostBreakdown {
  input_cost?: number;
  output_cost?: number;
  total_cost?: number;
  tool_usage_cost?: number;
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
}) => {
  if (!costBreakdown) {
    return null;
  }

  const hasDiscount =
    costBreakdown.discount_percent !== undefined ||
    costBreakdown.discount_amount !== undefined;
  const hasMargin =
    costBreakdown.margin_percent !== undefined ||
    costBreakdown.margin_fixed_amount !== undefined ||
    costBreakdown.margin_total_amount !== undefined;

  // Don't show if there's no meaningful breakdown data
  const hasMeaningfulData =
    costBreakdown.input_cost !== undefined ||
    costBreakdown.output_cost !== undefined ||
    hasDiscount ||
    hasMargin;

  if (!hasMeaningfulData) {
    return null;
  }

  return (
    <div className="bg-white rounded-lg shadow w-full max-w-full overflow-hidden">
      <Accordion>
        <AccordionHeader className="p-4 border-b hover:bg-gray-50 transition-colors">
          <div className="flex items-center justify-between w-full">
            <h3 className="text-lg font-medium text-gray-900">Cost Breakdown</h3>
            <div className="flex items-center space-x-2 mr-4">
              <span className="text-sm text-gray-500">Total:</span>
              <span className="text-sm font-semibold text-gray-900">{formatCost(totalSpend)}</span>
            </div>
          </div>
        </AccordionHeader>
        <AccordionBody className="px-0">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 p-4 w-full max-w-full overflow-hidden">
            {/* Left Column: Token Costs */}
            <div className="space-y-2">
              <div className="flex">
                <span className="font-medium w-1/3 text-gray-700 text-sm">Input Cost:</span>
                <span className="text-sm text-gray-900">{formatCost(costBreakdown.input_cost)}</span>
              </div>
              <div className="flex">
                <span className="font-medium w-1/3 text-gray-700 text-sm">Output Cost:</span>
                <span className="text-sm text-gray-900">{formatCost(costBreakdown.output_cost)}</span>
              </div>
              {costBreakdown.tool_usage_cost !== undefined &&
                costBreakdown.tool_usage_cost > 0 && (
                  <div className="flex">
                    <span className="font-medium w-1/3 text-gray-700 text-sm">Tool Usage Cost:</span>
                    <span className="text-sm text-gray-900">{formatCost(costBreakdown.tool_usage_cost)}</span>
                  </div>
                )}
            </div>

            {/* Right Column: Adjustments */}
            <div className="space-y-2">
              {hasDiscount && (
                <>
                  {costBreakdown.original_cost !== undefined && (
                    <div className="flex">
                      <span className="font-medium w-1/3 text-gray-700 text-sm">Original Cost:</span>
                      <span className="text-sm text-gray-900 font-medium">{formatCost(costBreakdown.original_cost)}</span>
                    </div>
                  )}
                  {costBreakdown.discount_percent !== undefined && (
                    <div className="flex">
                      <span className="font-medium w-1/3 text-gray-700 text-sm">Discount (%):</span>
                      <span className="text-sm text-gray-900">-{formatPercent(costBreakdown.discount_percent)}</span>
                    </div>
                  )}
                  {costBreakdown.discount_amount !== undefined && (
                    <div className="flex">
                      <span className="font-medium w-1/3 text-gray-700 text-sm">Discount Amount:</span>
                      <span className="text-sm text-gray-900">-{formatCost(costBreakdown.discount_amount)}</span>
                    </div>
                  )}
                </>
              )}

              {hasMargin && (
                <>
                  {costBreakdown.margin_percent !== undefined &&
                    costBreakdown.margin_percent > 0 && (
                      <div className="flex">
                        <span className="font-medium w-1/3 text-gray-700 text-sm">Margin (%):</span>
                        <span className="text-sm text-gray-900">+{formatPercent(costBreakdown.margin_percent)}</span>
                      </div>
                    )}
                  {costBreakdown.margin_fixed_amount !== undefined &&
                    costBreakdown.margin_fixed_amount > 0 && (
                      <div className="flex">
                        <span className="font-medium w-1/3 text-gray-700 text-sm">Fixed Margin:</span>
                        <span className="text-sm text-gray-900">+{formatCost(costBreakdown.margin_fixed_amount)}</span>
                      </div>
                    )}
                  {costBreakdown.margin_total_amount !== undefined &&
                    costBreakdown.margin_total_amount > 0 && (
                      <div className="flex">
                        <span className="font-medium w-1/3 text-gray-700 text-sm">Total Margin Added:</span>
                        <span className="text-sm text-gray-900 font-medium">+{formatCost(costBreakdown.margin_total_amount)}</span>
                      </div>
                    )}
                </>
              )}

              {!hasDiscount && !hasMargin && (
                <div className="flex italic text-gray-400 text-sm">
                  <span>No adjustments applied</span>
                </div>
              )}
            </div>
          </div>
          
          <div className="p-4 border-t">
            <div className="flex items-center justify-between">
              <span className="font-medium text-sm text-gray-900">Final Calculated Cost:</span>
              <span className="text-lg font-bold text-gray-900">
                {formatCost(costBreakdown.total_cost ?? totalSpend)}
              </span>
            </div>
          </div>
        </AccordionBody>
      </Accordion>
    </div>
  );
};

export default CostBreakdownViewer;
