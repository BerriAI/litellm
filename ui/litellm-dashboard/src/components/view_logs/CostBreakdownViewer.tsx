import React from "react";
import { Collapse } from "antd";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { deriveRate, splitPromptTokens } from "./cacheCostBreakdown";

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
  providerCacheReadTokens?: number;
}

const formatCost = (cost: number | undefined): string => {
  if (cost === undefined || cost === null) return "-";
  return `$${formatNumberWithCommas(cost, 8)}`;
};

const formatRate = (rate: number | undefined): string => {
  if (rate === undefined || rate === null || !Number.isFinite(rate)) return "-";
  // Per-token rates are tiny (e.g. 5.9e-7). Show up to 12 significant digits and
  // trim trailing zeros so 0.00000059 renders as "$0.00000059", not "$0.00000059000".
  return `$${Number(rate.toPrecision(12)).toLocaleString("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 20,
  })}`;
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
  providerCacheReadTokens,
}) => {
  const isCached = cacheHit?.toLowerCase() === "true";
  const hasTokenCounts = promptTokens !== undefined || completionTokens !== undefined;

  const hasCostBreakdown = costBreakdown?.input_cost !== undefined || costBreakdown?.output_cost !== undefined;
  const hasAdditionalCosts =
    costBreakdown?.additional_costs &&
    Object.entries(costBreakdown.additional_costs).some(([, value]) => value != null && value !== 0);
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
  const totalCost = isCached ? 0 : costBreakdown?.total_cost ?? totalSpend;

  // Provider prompt-cache formula breakdown (issue #32045). Never shown when the
  // LiteLLM response cache served the request (isCached) since costs are $0 there.
  const tokenSplit = splitPromptTokens(promptTokens ?? 0, providerCacheReadTokens ?? 0);
  const hasItemizedCacheCost =
    costBreakdown?.cache_read_cost !== undefined || costBreakdown?.cache_creation_cost !== undefined;
  const cacheReadCost = costBreakdown?.cache_read_cost;
  // Full-rate (cache-miss) input cost is the input cost minus the discounted
  // cache-read and premium cache-write portions. deriveRate rejects a negative
  // result, so inconsistent backend costs hide the row instead of showing a
  // negative rate.
  const cacheMissInputCost =
    inputCost !== undefined ? inputCost - (cacheReadCost ?? 0) - (costBreakdown?.cache_creation_cost ?? 0) : undefined;
  const cacheMissRate = deriveRate(cacheMissInputCost, tokenSplit.cacheMissTokens);
  const cacheReadRate = deriveRate(cacheReadCost, tokenSplit.cacheHitTokens);
  const outputRate = deriveRate(outputCost, completionTokens);
  // The per-token input formula is only accurate when cache_read_cost is known,
  // since otherwise input_cost blends the miss and hit rates and cannot be split.
  // Render it directly under the Input Cost line (inside the itemized branch) so
  // it attaches to the right row rather than floating below the whole section.
  const showInputCacheFormula =
    !isCached && tokenSplit.cacheHitTokens > 0 && cacheMissRate !== undefined && cacheReadRate !== undefined;
  const showProviderCacheFormula = !isCached && tokenSplit.cacheHitTokens > 0;

  const inputCacheFormula = showInputCacheFormula ? (
    <div className="pl-1 text-xs text-gray-500 font-mono space-y-0.5">
      <div>
        = {formatNumberWithCommas(tokenSplit.cacheMissTokens)} cache miss tokens * {formatRate(cacheMissRate)}
      </div>
      <div>
        + {formatNumberWithCommas(tokenSplit.cacheHitTokens)} cache hit tokens * {formatRate(cacheReadRate)}
      </div>
    </div>
  ) : null;

  return (
    <div className="bg-white rounded-lg shadow-sm w-full max-w-full overflow-hidden mb-6">
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
                  {(() => {
                    if (hasItemizedCacheCost) {
                      // Separate line items: Input / Cache Read / Cache Write
                      const rawCost = isCached
                        ? 0
                        : (inputCost ?? 0) -
                          (costBreakdown?.cache_read_cost ?? 0) -
                          (costBreakdown?.cache_creation_cost ?? 0);
                      return (
                        <>
                          <div className="flex text-sm">
                            <span className="text-gray-600 font-medium w-1/3">Input Cost:</span>
                            <span className="text-gray-900">
                              {formatCost(rawCost)}
                              {rawInputTokens !== undefined && rawInputTokens !== null && (
                                <span className="text-gray-500 font-normal ml-1">
                                  ({rawInputTokens.toLocaleString()} tokens)
                                </span>
                              )}
                            </span>
                          </div>
                          {inputCacheFormula}
                          {(costBreakdown?.cache_read_cost ?? 0) > 0 && (
                            <div className="flex text-sm">
                              <span className="text-gray-600 font-medium w-1/3">Prompt Cache Read Cost:</span>
                              <span className="text-gray-900">
                                {formatCost(isCached ? 0 : costBreakdown?.cache_read_cost)}
                                {(cacheReadTokens ?? 0) > 0 && (
                                  <span className="text-gray-500 font-normal ml-1">
                                    ({(cacheReadTokens ?? 0).toLocaleString()} tokens)
                                  </span>
                                )}
                              </span>
                            </div>
                          )}
                          {(costBreakdown?.cache_creation_cost ?? 0) > 0 && (
                            <div className="flex text-sm">
                              <span className="text-gray-600 font-medium w-1/3">Prompt Cache Write Cost:</span>
                              <span className="text-gray-900">
                                {formatCost(isCached ? 0 : costBreakdown?.cache_creation_cost)}
                                {(cacheCreationTokens ?? 0) > 0 && (
                                  <span className="text-gray-500 font-normal ml-1">
                                    ({(cacheCreationTokens ?? 0).toLocaleString()} tokens)
                                  </span>
                                )}
                              </span>
                            </div>
                          )}
                        </>
                      );
                    }
                    return (
                      <>
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
                        {inputCacheFormula}
                      </>
                    );
                  })()}
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
                  {showProviderCacheFormula && outputRate !== undefined && (
                    <div className="pl-1 text-xs text-gray-500 font-mono">
                      = {formatNumberWithCommas(completionTokens ?? 0)} completion tokens * {formatRate(outputRate)}
                    </div>
                  )}
                  {costBreakdown?.tool_usage_cost !== undefined && costBreakdown.tool_usage_cost > 0 && (
                    <div className="flex text-sm">
                      <span className="text-gray-600 font-medium w-1/3">Tool Usage Cost:</span>
                      <span className="text-gray-900">{formatCost(costBreakdown.tool_usage_cost)}</span>
                    </div>
                  )}
                  {costBreakdown?.additional_costs &&
                    Object.entries(costBreakdown.additional_costs)
                      .filter(([, value]) => value != null && value !== 0)
                      .map(([key, value]) => (
                        <div key={key} className="flex text-sm">
                          <span className="text-gray-600 font-medium w-1/3">{key}:</span>
                          <span className="text-gray-900">{formatCost(value)}</span>
                        </div>
                      ))}
                </div>

                {/* Subtotal / Original Cost - hide when cached since it would be $0 */}
                {!isCached && (
                  <div className="pt-2 border-t border-gray-100 max-w-2xl">
                    <div className="flex text-sm font-semibold">
                      <span className="text-gray-900 w-1/3">Original LLM Cost:</span>
                      <span className="text-gray-900">{formatCost(originalCost)}</span>
                    </div>
                    {showProviderCacheFormula && inputCost !== undefined && outputCost !== undefined && (
                      <div className="pl-1 mt-0.5 text-xs text-gray-500 font-mono">
                        = {formatCost(inputCost)} input cost + {formatCost(outputCost)} output cost
                      </div>
                    )}
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
                            <span className="font-medium w-1/3">
                              Discount ({formatPercent(costBreakdown.discount_percent)}):
                            </span>
                            <span className="text-gray-900">-{formatCost(costBreakdown.discount_amount)}</span>
                          </div>
                        )}
                        {costBreakdown.discount_amount !== undefined &&
                          costBreakdown.discount_percent === undefined && (
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
                            <span className="font-medium w-1/3">
                              Margin ({formatPercent(costBreakdown.margin_percent)}):
                            </span>
                            <span className="text-gray-900">
                              +
                              {formatCost(
                                (costBreakdown.margin_total_amount || 0) - (costBreakdown.margin_fixed_amount || 0),
                              )}
                            </span>
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
