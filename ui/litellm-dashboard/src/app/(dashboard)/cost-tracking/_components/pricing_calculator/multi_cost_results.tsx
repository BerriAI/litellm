import React, { useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { CostEstimateResponse } from "../types";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { MultiModelResult } from "./types";
import MultiExportDropdown from "./multi_export_dropdown";

interface MultiCostResultsProps {
  multiResult: MultiModelResult;
  timePeriod: "day" | "month";
}

const formatCost = (value: number | null | undefined): string => {
  if (value === null || value === undefined) return "-";
  if (value === 0) return "$0";
  if (value < 0.0001) return `$${value.toExponential(2)}`;
  if (value < 1) return `$${value.toFixed(4)}`;
  return `$${formatNumberWithCommas(value, 2, true)}`;
};

const formatRequests = (value: number | null | undefined): string => {
  if (value === null || value === undefined) return "-";
  return formatNumberWithCommas(value, 0, true);
};

const SingleModelBreakdown: React.FC<{
  result: CostEstimateResponse;
  loading: boolean;
  timePeriod: "day" | "month";
}> = ({ result, loading, timePeriod }) => {
  const periodLabel = timePeriod === "day" ? "Daily" : "Monthly";
  const periodCost = timePeriod === "day" ? result.daily_cost : result.monthly_cost;
  const periodInputCost = timePeriod === "day" ? result.daily_input_cost : result.monthly_input_cost;
  const periodOutputCost = timePeriod === "day" ? result.daily_output_cost : result.monthly_output_cost;
  const periodMarginCost = timePeriod === "day" ? result.daily_margin_cost : result.monthly_margin_cost;
  const periodRequests = timePeriod === "day" ? result.num_requests_per_day : result.num_requests_per_month;

  return (
    <div className="space-y-3 bg-muted p-4 rounded-lg">
      {loading && (
        <div className="flex items-center gap-2 text-muted-foreground text-sm">
          <UiLoadingSpinner className="size-3.5" />
          <span>Updating...</span>
        </div>
      )}

      <div className="grid grid-cols-4 gap-4">
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground block">Total/Request</p>
          <p className="text-base font-semibold text-info break-words">{formatCost(result.cost_per_request)}</p>
        </div>
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground block">Input Cost</p>
          <p className="text-sm break-words">{formatCost(result.input_cost_per_request)}</p>
        </div>
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground block">Output Cost</p>
          <p className="text-sm break-words">{formatCost(result.output_cost_per_request)}</p>
        </div>
        <div className="min-w-0">
          <p className="text-xs text-muted-foreground block">Margin Fee</p>
          <p className={`text-sm break-words ${result.margin_cost_per_request > 0 ? "text-warning" : ""}`}>
            {formatCost(result.margin_cost_per_request)}
          </p>
        </div>
      </div>

      {periodCost !== null && (
        <div className="grid grid-cols-4 gap-4 pt-2 border-t border-border">
          <div className="min-w-0">
            <p className="text-xs text-muted-foreground block">
              {periodLabel} Total ({formatRequests(periodRequests)} req)
            </p>
            <p
              className={`text-base font-semibold break-words ${timePeriod === "day" ? "text-success" : "text-purple-600 dark:text-purple-300"}`}
            >
              {formatCost(periodCost)}
            </p>
          </div>
          <div className="min-w-0">
            <p className="text-xs text-muted-foreground block">{periodLabel} Input</p>
            <p className="text-sm break-words">{formatCost(periodInputCost)}</p>
          </div>
          <div className="min-w-0">
            <p className="text-xs text-muted-foreground block">{periodLabel} Output</p>
            <p className="text-sm break-words">{formatCost(periodOutputCost)}</p>
          </div>
          <div className="min-w-0">
            <p className="text-xs text-muted-foreground block">{periodLabel} Margin Fee</p>
            <p className={`text-sm break-words ${(periodMarginCost ?? 0) > 0 ? "text-warning" : ""}`}>
              {formatCost(periodMarginCost)}
            </p>
          </div>
        </div>
      )}

      {(result.input_cost_per_token || result.output_cost_per_token) && (
        <div className="text-xs text-muted-foreground pt-2 border-t border-border">
          Token Pricing:{" "}
          {result.input_cost_per_token && (
            <span>Input ${formatNumberWithCommas(result.input_cost_per_token * 1_000_000, 2)}/1M</span>
          )}
          {result.input_cost_per_token && result.output_cost_per_token && " | "}
          {result.output_cost_per_token && (
            <span>Output ${formatNumberWithCommas(result.output_cost_per_token * 1_000_000, 2)}/1M</span>
          )}
        </div>
      )}
    </div>
  );
};

const MultiCostResults: React.FC<MultiCostResultsProps> = ({ multiResult, timePeriod }) => {
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());

  const validEntries = multiResult.entries.filter((e) => e.result !== null);
  const loadingEntries = multiResult.entries.filter((e) => e.loading);
  const errorEntries = multiResult.entries.filter((e) => e.error !== null);
  const hasAnyResult = validEntries.length > 0;
  const isAnyLoading = loadingEntries.length > 0;
  const hasAnyError = errorEntries.length > 0;

  // Show empty state only if no results, not loading, and no errors
  if (!hasAnyResult && !isAnyLoading && !hasAnyError) {
    return (
      <div className="py-6 text-center border border-dashed border-border rounded-lg bg-muted">
        <p className="text-muted-foreground">Select models above to see cost estimates</p>
      </div>
    );
  }

  // Show loading state only if loading and no results/errors yet
  if (!hasAnyResult && isAnyLoading && !hasAnyError) {
    return (
      <div className="py-6 text-center">
        <UiLoadingSpinner className="inline-block size-5" />
        <p className="text-muted-foreground block mt-2">Calculating costs...</p>
      </div>
    );
  }

  // Show errors-only view when there are errors but no valid results
  if (!hasAnyResult && hasAnyError) {
    return (
      <div className="space-y-4">
        <Separator className="my-4" />
        <div className="flex items-center justify-between">
          <p className="text-base font-semibold text-foreground">Cost Estimates</p>
          {isAnyLoading && <UiLoadingSpinner className="size-3.5" />}
        </div>
        {/* Error Messages */}
        {errorEntries.map((e) => (
          <div
            key={e.entry.id}
            className="text-sm text-destructive bg-destructive/10 p-3 rounded-lg border border-destructive/20"
          >
            <span className="font-medium">{e.entry.model || "Unknown model"}: </span>
            {e.error}
          </div>
        ))}
      </div>
    );
  }

  const toggleExpanded = (id: string) => {
    setExpandedModels((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const hasMargin = multiResult.totals.margin_per_request > 0;

  const periodLabel = timePeriod === "day" ? "Daily" : "Monthly";

  // Include both valid results and errors in the table data
  const allEntriesWithModels = multiResult.entries.filter((e) => e.entry.model);
  const summaryData = allEntriesWithModels.map((e) => ({
    id: e.entry.id,
    model: e.result?.model || e.entry.model,
    provider: e.result?.provider,
    cost_per_request: e.result?.cost_per_request ?? null,
    margin_cost_per_request: e.result?.margin_cost_per_request ?? null,
    daily_cost: e.result?.daily_cost ?? null,
    monthly_cost: e.result?.monthly_cost ?? null,
    error: e.error,
    loading: e.loading,
    hasZeroCost: e.result && e.result.cost_per_request === 0,
  }));

  return (
    <div className="space-y-4">
      <Separator className="my-4" />

      <div className="flex items-center justify-between">
        <p className="text-base font-semibold text-foreground">Cost Estimates</p>
        <div className="flex items-center gap-2">
          {isAnyLoading && <UiLoadingSpinner className="size-3.5" />}
          <MultiExportDropdown multiResult={multiResult} />
        </div>
      </div>

      {/* Combined Totals - Always show when there are results */}
      <Card size="sm" className="px-4 bg-linear-to-r from-slate-50 to-blue-50 dark:from-slate-900 dark:to-blue-950">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-2">
          <div className="min-w-0">
            <span className="text-xs text-muted-foreground">Total Per Request</span>
            <div className="text-lg font-mono text-info break-words">
              {formatCost(multiResult.totals.cost_per_request)}
            </div>
          </div>
          <div className="min-w-0">
            <span className="text-xs text-muted-foreground">Total {periodLabel}</span>
            <div
              className={`text-lg font-mono break-words ${timePeriod === "day" ? "text-success" : "text-purple-600 dark:text-purple-300"}`}
            >
              {formatCost(timePeriod === "day" ? multiResult.totals.daily_cost : multiResult.totals.monthly_cost)}
            </div>
          </div>
        </div>
        {hasMargin && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-x-4 gap-y-2 mt-3 pt-3 border-t border-border">
            <div className="min-w-0">
              <div className="text-xs text-muted-foreground">Margin Fee/Request</div>
              <div className="text-sm font-mono text-warning break-words">
                {formatCost(multiResult.totals.margin_per_request)}
              </div>
            </div>
            <div className="min-w-0">
              <div className="text-xs text-muted-foreground">{periodLabel} Margin Fee</div>
              <div className="text-sm font-mono text-warning break-words">
                {formatCost(timePeriod === "day" ? multiResult.totals.daily_margin : multiResult.totals.monthly_margin)}
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* Per-Model Table */}
      {summaryData.length > 0 && (
        <Table className="border border-border rounded-lg">
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">Per Request</TableHead>
              <TableHead className="text-right">Margin Fee</TableHead>
              <TableHead className="text-right">{periodLabel}</TableHead>
              <TableHead className="w-10">
                <span className="sr-only">Cost breakdown</span>
              </TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {summaryData.map((record) => {
              const isExpanded = expandedModels.has(record.id);
              const periodCost = timePeriod === "day" ? record.daily_cost : record.monthly_cost;
              const breakdownEntry = validEntries.find((e) => e.entry.id === record.id);
              return (
                <React.Fragment key={record.id}>
                  <TableRow>
                    <TableCell className="whitespace-normal">
                      <div className="flex min-w-0 flex-col gap-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium text-sm break-words">{record.model}</span>
                          {record.provider && (
                            <Badge variant="secondary" className="text-xs">
                              {record.provider}
                            </Badge>
                          )}
                          {record.loading && <UiLoadingSpinner className="size-3.5" />}
                        </div>
                        {record.error && (
                          <div className="text-xs text-destructive bg-destructive/10 px-2 py-1 rounded-sm">
                            ⚠️ {record.error}
                          </div>
                        )}
                        {record.hasZeroCost && !record.error && (
                          <div className="text-xs text-warning bg-warning/10 px-2 py-1 rounded-sm">
                            ⚠️ No pricing data found for this model. Set base_model in config.
                          </div>
                        )}
                      </div>
                    </TableCell>
                    <TableCell className="text-right">
                      {record.error ? (
                        <span className="text-muted-foreground">-</span>
                      ) : (
                        <span className="font-mono text-sm">{formatCost(record.cost_per_request)}</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {record.error ? (
                        <span className="text-muted-foreground">-</span>
                      ) : (
                        <span
                          className={`font-mono text-sm ${(record.margin_cost_per_request ?? 0) > 0 ? "text-warning" : "text-muted-foreground"}`}
                        >
                          {formatCost(record.margin_cost_per_request)}
                        </span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {record.error ? (
                        <span className="text-muted-foreground">-</span>
                      ) : (
                        <span className="font-mono text-sm">{formatCost(periodCost)}</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      {!record.error && (
                        <Button
                          variant="ghost"
                          size="icon-xs"
                          aria-expanded={isExpanded}
                          aria-label={`${isExpanded ? "Hide" : "Show"} cost breakdown for ${record.model}`}
                          onClick={() => toggleExpanded(record.id)}
                          className="text-muted-foreground hover:text-foreground"
                        >
                          {isExpanded ? <ChevronDown className="size-3" /> : <ChevronRight className="size-3" />}
                        </Button>
                      )}
                    </TableCell>
                  </TableRow>
                  {isExpanded && breakdownEntry?.result && (
                    <TableRow>
                      <TableCell colSpan={5} className="whitespace-normal">
                        <div className="py-2">
                          <SingleModelBreakdown
                            result={breakdownEntry.result}
                            loading={breakdownEntry.loading}
                            timePeriod={timePeriod}
                          />
                        </div>
                      </TableCell>
                    </TableRow>
                  )}
                </React.Fragment>
              );
            })}
          </TableBody>
        </Table>
      )}
    </div>
  );
};

export default MultiCostResults;
