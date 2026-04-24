import React from "react";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export interface CostBreakdown {
  input_cost?: number;
  cache_read_cost?: number;
  cache_creation_cost?: number;
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
  rawInputTokens?: number;
  cacheReadTokens?: number;
  cacheCreationTokens?: number;
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
  rawInputTokens,
  cacheReadTokens,
  cacheCreationTokens,
}) => {
  const isCached = cacheHit?.toLowerCase() === "true";
  const hasTokenCounts = promptTokens !== undefined || completionTokens !== undefined;

  const hasCostBreakdown = costBreakdown?.input_cost !== undefined || costBreakdown?.output_cost !== undefined;
  const hasAdditionalCosts =
    costBreakdown?.additional_costs &&
    Object.entries(costBreakdown.additional_costs).some(
      ([, value]) => value != null && value !== 0
    );
  const hasMeaningfulData =
    hasCostBreakdown ||
    hasTokenCounts ||
    hasAdditionalCosts ||
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
    <div className="bg-background rounded-lg shadow w-full max-w-full overflow-hidden mb-6">
      <Accordion type="single" collapsible className="w-full">
        <AccordionItem value="cost-breakdown" className="border-b-0">
          <AccordionTrigger className="px-4 py-3 hover:no-underline">
            <div className="flex items-center justify-between w-full pr-4">
              <h3 className="text-lg font-medium text-foreground">Cost Breakdown</h3>
              <div className="flex items-center space-x-2 mr-4">
                <span className="text-sm text-muted-foreground">Total:</span>
                <span className="text-sm font-semibold text-foreground">
                  {formatCost(totalSpend)}
                  {isCached && " (Cached)"}
                </span>
              </div>
            </div>
          </AccordionTrigger>
          <AccordionContent>
            <div className="p-6 space-y-4">
              {/* Step 1: Base Token Costs */}
              <div className="space-y-2 max-w-2xl">
                {(() => {
                  const hasCacheBreakdown =
                    costBreakdown?.cache_read_cost !== undefined ||
                    costBreakdown?.cache_creation_cost !== undefined;
                  if (hasCacheBreakdown) {
                    // Separate line items: Input / Cache Read / Cache Write
                    const rawCost = isCached ? 0 : (inputCost ?? 0) - (costBreakdown?.cache_read_cost ?? 0) - (costBreakdown?.cache_creation_cost ?? 0);
                    return (
                      <>
                        <div className="flex text-sm">
                          <span className="text-muted-foreground font-medium w-1/3">Input Cost:</span>
                          <span className="text-foreground">
                            {formatCost(rawCost)}
                            {rawInputTokens !== undefined && rawInputTokens !== null && (
                              <span className="text-muted-foreground font-normal ml-1">({rawInputTokens.toLocaleString()} tokens)</span>
                            )}
                          </span>
                        </div>
                        {(costBreakdown?.cache_read_cost ?? 0) > 0 && (
                          <div className="flex text-sm">
                            <span className="text-muted-foreground font-medium w-1/3">Cache Read Cost:</span>
                            <span className="text-foreground">
                              {formatCost(isCached ? 0 : costBreakdown?.cache_read_cost)}
                              {(cacheReadTokens ?? 0) > 0 && (
                                <span className="text-muted-foreground font-normal ml-1">({(cacheReadTokens ?? 0).toLocaleString()} tokens)</span>
                              )}
                            </span>
                          </div>
                        )}
                        {(costBreakdown?.cache_creation_cost ?? 0) > 0 && (
                          <div className="flex text-sm">
                            <span className="text-muted-foreground font-medium w-1/3">Cache Write Cost:</span>
                            <span className="text-foreground">
                              {formatCost(isCached ? 0 : costBreakdown?.cache_creation_cost)}
                              {(cacheCreationTokens ?? 0) > 0 && (
                                <span className="text-muted-foreground font-normal ml-1">({(cacheCreationTokens ?? 0).toLocaleString()} tokens)</span>
                              )}
                            </span>
                          </div>
                        )}
                      </>
                    );
                  }
                  return (
                    <div className="flex text-sm">
                      <span className="text-muted-foreground font-medium w-1/3">Input Cost:</span>
                      <span className="text-foreground">
                        {formatCost(inputCost)}
                        {promptTokens !== undefined && (
                          <span className="text-muted-foreground font-normal ml-1">
                            ({promptTokens.toLocaleString()} prompt tokens)
                          </span>
                        )}
                      </span>
                    </div>
                  );
                })()}
                <div className="flex text-sm">
                  <span className="text-muted-foreground font-medium w-1/3">Output Cost:</span>
                  <span className="text-foreground">
                    {formatCost(outputCost)}
                    {completionTokens !== undefined && (
                      <span className="text-muted-foreground font-normal ml-1">
                        ({completionTokens.toLocaleString()} completion tokens)
                      </span>
                    )}
                  </span>
                </div>
                {costBreakdown?.tool_usage_cost !== undefined && costBreakdown.tool_usage_cost > 0 && (
                  <div className="flex text-sm">
                    <span className="text-muted-foreground font-medium w-1/3">Tool Usage Cost:</span>
                    <span className="text-foreground">{formatCost(costBreakdown.tool_usage_cost)}</span>
                  </div>
                )}
                {costBreakdown?.additional_costs &&
                  Object.entries(costBreakdown.additional_costs)
                    .filter(([, value]) => value != null && value !== 0)
                    .map(([key, value]) => (
                      <div key={key} className="flex text-sm">
                        <span className="text-muted-foreground font-medium w-1/3">{key}:</span>
                        <span className="text-foreground">{formatCost(value)}</span>
                      </div>
                    ))}
              </div>

              {/* Subtotal / Original Cost - hide when cached since it would be $0 */}
              {!isCached && (
                <div className="pt-2 border-t border-border max-w-2xl">
                  <div className="flex text-sm font-semibold">
                    <span className="text-foreground w-1/3">Original LLM Cost:</span>
                    <span className="text-foreground">{formatCost(originalCost)}</span>
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
                        <div className="flex text-sm text-muted-foreground">
                          <span className="font-medium w-1/3">Discount ({formatPercent(costBreakdown.discount_percent)}):</span>
                          <span className="text-foreground">-{formatCost(costBreakdown.discount_amount)}</span>
                        </div>
                      )}
                      {costBreakdown.discount_amount !== undefined && costBreakdown.discount_percent === undefined && (
                        <div className="flex text-sm text-muted-foreground">
                          <span className="font-medium w-1/3">Discount Amount:</span>
                          <span className="text-foreground">-{formatCost(costBreakdown.discount_amount)}</span>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Margins */}
                  {hasMargin && (
                    <div className="space-y-2">
                      {costBreakdown.margin_percent !== undefined && costBreakdown.margin_percent !== 0 && (
                        <div className="flex text-sm text-muted-foreground">
                          <span className="font-medium w-1/3">Margin ({formatPercent(costBreakdown.margin_percent)}):</span>
                          <span className="text-foreground">+{formatCost((costBreakdown.margin_total_amount || 0) - (costBreakdown.margin_fixed_amount || 0))}</span>
                        </div>
                      )}
                      {costBreakdown.margin_fixed_amount !== undefined && costBreakdown.margin_fixed_amount !== 0 && (
                        <div className="flex text-sm text-muted-foreground">
                          <span className="font-medium w-1/3">Margin:</span>
                          <span className="text-foreground">+{formatCost(costBreakdown.margin_fixed_amount)}</span>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Final Summary */}
              <div className="mt-4 pt-4 border-t border-border max-w-2xl">
                <div className="flex items-center">
                  <span className="font-bold text-sm text-foreground w-1/3">Final Calculated Cost:</span>
                  <span className="text-sm font-bold text-foreground">
                    {formatCost(totalCost)}
                    {isCached && " (Cached)"}
                  </span>
                </div>
              </div>
            </div>
          </AccordionContent>
        </AccordionItem>
      </Accordion>
    </div>
  );
};

export default CostBreakdownViewer;
