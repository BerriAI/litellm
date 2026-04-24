import React, { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { ChevronDown, ChevronRight, LoaderCircle } from "lucide-react";
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

const Spinner: React.FC<{ size?: "sm" | "md" }> = ({ size = "md" }) => (
  <LoaderCircle className={`${size === "sm" ? "h-3.5 w-3.5" : "h-5 w-5"} animate-spin text-muted-foreground`} />
);

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
          <Spinner size="sm" />
          <span>Updating...</span>
        </div>
      )}

      <div className="grid grid-cols-4 gap-4">
        <div>
          <span className="text-xs text-muted-foreground block">Total/Request</span>
          <span className="text-base font-semibold text-blue-600 dark:text-blue-400">
            {formatCost(result.cost_per_request)}
          </span>
        </div>
        <div>
          <span className="text-xs text-muted-foreground block">Input Cost</span>
          <span className="text-sm">{formatCost(result.input_cost_per_request)}</span>
        </div>
        <div>
          <span className="text-xs text-muted-foreground block">Output Cost</span>
          <span className="text-sm">{formatCost(result.output_cost_per_request)}</span>
        </div>
        <div>
          <span className="text-xs text-muted-foreground block">Margin Fee</span>
          <span className={`text-sm ${result.margin_cost_per_request > 0 ? "text-amber-600 dark:text-amber-400" : ""}`}>
            {formatCost(result.margin_cost_per_request)}
          </span>
        </div>
      </div>

      {periodCost !== null && (
        <div className="grid grid-cols-4 gap-4 pt-2 border-t border-border">
          <div>
            <span className="text-xs text-muted-foreground block">
              {periodLabel} Total ({formatRequests(periodRequests)} req)
            </span>
            <span
              className={`text-base font-semibold ${
                timePeriod === "day" ? "text-green-600 dark:text-green-400" : "text-purple-600 dark:text-purple-400"
              }`}
            >
              {formatCost(periodCost)}
            </span>
          </div>
          <div>
            <span className="text-xs text-muted-foreground block">{periodLabel} Input</span>
            <span className="text-sm">{formatCost(periodInputCost)}</span>
          </div>
          <div>
            <span className="text-xs text-muted-foreground block">{periodLabel} Output</span>
            <span className="text-sm">{formatCost(periodOutputCost)}</span>
          </div>
          <div>
            <span className="text-xs text-muted-foreground block">{periodLabel} Margin Fee</span>
            <span className={`text-sm ${(periodMarginCost ?? 0) > 0 ? "text-amber-600 dark:text-amber-400" : ""}`}>
              {formatCost(periodMarginCost)}
            </span>
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

interface SummaryRow {
  id: string;
  model: string;
  provider?: string | null;
  cost_per_request: number | null;
  margin_cost_per_request: number | null;
  daily_cost: number | null;
  monthly_cost: number | null;
  error: string | null;
  loading: boolean;
  hasZeroCost: boolean | null;
}

const MultiCostResults: React.FC<MultiCostResultsProps> = ({ multiResult, timePeriod }) => {
  const [expandedModels, setExpandedModels] = useState<Set<string>>(new Set());

  const validEntries = multiResult.entries.filter((e) => e.result !== null);
  const loadingEntries = multiResult.entries.filter((e) => e.loading);
  const errorEntries = multiResult.entries.filter((e) => e.error !== null);
  const hasAnyResult = validEntries.length > 0;
  const isAnyLoading = loadingEntries.length > 0;
  const hasAnyError = errorEntries.length > 0;

  if (!hasAnyResult && !isAnyLoading && !hasAnyError) {
    return (
      <div className="py-6 text-center border border-dashed border-border rounded-lg bg-muted">
        <span className="text-muted-foreground">Select models above to see cost estimates</span>
      </div>
    );
  }

  if (!hasAnyResult && isAnyLoading && !hasAnyError) {
    return (
      <div className="py-6 text-center flex flex-col items-center gap-2">
        <Spinner />
        <span className="text-muted-foreground">Calculating costs...</span>
      </div>
    );
  }

  if (!hasAnyResult && hasAnyError) {
    return (
      <div className="space-y-4">
        <Separator className="my-4" />
        <div className="flex items-center justify-between">
          <span className="text-base font-semibold text-foreground">Cost Estimates</span>
          {isAnyLoading && <Spinner size="sm" />}
        </div>
        {errorEntries.map((e) => (
          <div
            key={e.entry.id}
            className="text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/30 p-3 rounded-lg border border-red-200 dark:border-red-800"
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

  const allEntriesWithModels = multiResult.entries.filter((e) => e.entry.model);
  const summaryData: SummaryRow[] = allEntriesWithModels.map((e) => ({
    id: e.entry.id,
    model: e.result?.model || e.entry.model,
    provider: e.result?.provider,
    cost_per_request: e.result?.cost_per_request ?? null,
    margin_cost_per_request: e.result?.margin_cost_per_request ?? null,
    daily_cost: e.result?.daily_cost ?? null,
    monthly_cost: e.result?.monthly_cost ?? null,
    error: e.error,
    loading: e.loading,
    hasZeroCost: e.result ? e.result.cost_per_request === 0 : null,
  }));

  const periodTotal = timePeriod === "day" ? multiResult.totals.daily_cost : multiResult.totals.monthly_cost;
  const periodMargin = timePeriod === "day" ? multiResult.totals.daily_margin : multiResult.totals.monthly_margin;

  return (
    <div className="space-y-4">
      <Separator className="my-4" />

      <div className="flex items-center justify-between">
        <span className="text-base font-semibold text-foreground">Cost Estimates</span>
        <div className="flex items-center gap-2">
          {isAnyLoading && <Spinner size="sm" />}
          <MultiExportDropdown multiResult={multiResult} />
        </div>
      </div>

      <Card className="bg-gradient-to-r from-slate-50 to-blue-50 dark:from-slate-900 dark:to-blue-950 border-border p-4">
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-muted-foreground">Total Per Request</div>
            <div className="font-mono text-lg text-blue-600 dark:text-blue-400">
              {formatCost(multiResult.totals.cost_per_request)}
            </div>
          </div>
          <div>
            <div className="text-xs text-muted-foreground">Total {periodLabel}</div>
            <div
              className={`font-mono text-lg ${
                timePeriod === "day" ? "text-green-600 dark:text-green-400" : "text-purple-600 dark:text-purple-400"
              }`}
            >
              {formatCost(periodTotal)}
            </div>
          </div>
        </div>
        {hasMargin && (
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-3 pt-3 border-t border-border">
            <div>
              <div className="text-xs text-muted-foreground">Margin Fee/Request</div>
              <div className="text-sm font-mono text-amber-600 dark:text-amber-400">
                {formatCost(multiResult.totals.margin_per_request)}
              </div>
            </div>
            <div>
              <div className="text-xs text-muted-foreground">{periodLabel} Margin Fee</div>
              <div className="text-sm font-mono text-amber-600 dark:text-amber-400">{formatCost(periodMargin)}</div>
            </div>
          </div>
        )}
      </Card>

      {summaryData.length > 0 && (
        <div className="border border-border rounded-lg overflow-hidden">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="h-10">Model</TableHead>
                <TableHead className="h-10 text-right">Per Request</TableHead>
                <TableHead className="h-10 text-right">Margin Fee</TableHead>
                <TableHead className="h-10 text-right">{periodLabel}</TableHead>
                <TableHead className="h-10 w-[40px]" />
              </TableRow>
            </TableHeader>
            <TableBody>
              {summaryData.map((record) => {
                const isExpanded = expandedModels.has(record.id);
                const periodCost = timePeriod === "day" ? record.daily_cost : record.monthly_cost;
                return (
                  <React.Fragment key={record.id}>
                    <TableRow>
                      <TableCell className="py-2">
                        <div className="flex flex-col gap-1">
                          <div className="flex items-center gap-2">
                            <span className="font-medium text-sm">{record.model}</span>
                            {record.provider && (
                              <Badge variant="secondary" className="text-xs">
                                {record.provider}
                              </Badge>
                            )}
                            {record.loading && <Spinner size="sm" />}
                          </div>
                          {record.error && (
                            <div className="text-xs text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-950/30 px-2 py-1 rounded">
                              ⚠️ {record.error}
                            </div>
                          )}
                          {record.hasZeroCost && !record.error && (
                            <div className="text-xs text-amber-600 dark:text-amber-400 bg-amber-50 dark:bg-amber-950/30 px-2 py-1 rounded">
                              ⚠️ No pricing data found for this model. Set base_model in config.
                            </div>
                          )}
                        </div>
                      </TableCell>
                      <TableCell className="py-2 text-right">
                        {record.error ? (
                          <span className="text-muted-foreground">-</span>
                        ) : (
                          <span className="font-mono text-sm">{formatCost(record.cost_per_request)}</span>
                        )}
                      </TableCell>
                      <TableCell className="py-2 text-right">
                        {record.error ? (
                          <span className="text-muted-foreground">-</span>
                        ) : (
                          <span
                            className={`font-mono text-sm ${
                              (record.margin_cost_per_request ?? 0) > 0
                                ? "text-amber-600 dark:text-amber-400"
                                : "text-muted-foreground"
                            }`}
                          >
                            {formatCost(record.margin_cost_per_request)}
                          </span>
                        )}
                      </TableCell>
                      <TableCell className="py-2 text-right">
                        {record.error ? (
                          <span className="text-muted-foreground">-</span>
                        ) : (
                          <span className="font-mono text-sm">{formatCost(periodCost)}</span>
                        )}
                      </TableCell>
                      <TableCell className="py-2 w-[40px]">
                        {record.error ? null : (
                          <Button
                            variant="ghost"
                            size="icon"
                            className="h-6 w-6 text-muted-foreground hover:text-foreground"
                            onClick={() => toggleExpanded(record.id)}
                            aria-label={isExpanded ? "Collapse" : "Expand"}
                          >
                            {isExpanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
                          </Button>
                        )}
                      </TableCell>
                    </TableRow>
                    {isExpanded && !record.error && (
                      <TableRow>
                        <TableCell colSpan={5} className="py-2 bg-muted/50">
                          {(() => {
                            const entry = validEntries.find((e) => e.entry.id === record.id);
                            if (!entry?.result) return null;
                            return (
                              <SingleModelBreakdown
                                result={entry.result}
                                loading={entry.loading}
                                timePeriod={timePeriod}
                              />
                            );
                          })()}
                        </TableCell>
                      </TableRow>
                    )}
                  </React.Fragment>
                );
              })}
            </TableBody>
          </Table>
        </div>
      )}
    </div>
  );
};

export default MultiCostResults;
