"use client";

import React from "react";
import { X } from "lucide-react";
import { BarChart } from "@/components/shared/charts";
import type { ChartTooltipProps } from "@/components/shared/charts/chart_tooltip";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { components } from "@/lib/http/schema";

export type CacheActivityErrorBucket = components["schemas"]["CacheActivityErrorBucket"];

export const FAILED_REQUESTS_SERIES = "Failed requests";

export type ErrorClassCount = {
  error_class: string;
  count: number;
};

export type ErrorCodeDatum = {
  error_code: string;
  [FAILED_REQUESTS_SERIES]: number;
  classes: ErrorClassCount[];
};

export const groupErrorBuckets = (buckets: readonly CacheActivityErrorBucket[], callType: string): ErrorCodeDatum[] => {
  const rows = buckets.filter((bucket) => bucket.call_type === callType);
  return [...new Set(rows.map((row) => row.error_code))]
    .map((errorCode) => {
      const codeRows = rows.filter((row) => row.error_code === errorCode);
      return {
        error_code: errorCode,
        [FAILED_REQUESTS_SERIES]: codeRows.reduce((total, row) => total + row.count, 0),
        classes: codeRows
          .map((row) => ({ error_class: row.error_class, count: row.count }))
          .sort((a, b) => b.count - a.count),
      };
    })
    .sort((a, b) => b[FAILED_REQUESTS_SERIES] - a[FAILED_REQUESTS_SERIES]);
};

export const ErrorCodeTooltip = ({ active, payload, label }: ChartTooltipProps) => {
  if (!active || !payload || payload.length === 0) return null;
  const datum = payload[0]?.payload as ErrorCodeDatum | undefined;
  if (!datum) return null;

  return (
    <div className="min-w-40 rounded-lg border border-border/50 bg-background px-2.5 py-1.5 text-xs shadow-xl">
      <p className="mb-1.5 font-medium text-foreground">
        Error code {String(label)}: {datum[FAILED_REQUESTS_SERIES].toLocaleString()} failed
      </p>
      <div className="grid gap-1.5">
        {datum.classes.map((errorClass) => (
          <div key={errorClass.error_class} className="flex w-full items-center justify-between gap-4">
            <span className="text-muted-foreground">{errorClass.error_class}</span>
            <span className="font-mono font-medium tabular-nums text-foreground">
              {errorClass.count.toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
};

interface ErrorDrilldownCardProps {
  callType: string;
  buckets: readonly CacheActivityErrorBucket[];
  valueFormatter: (value: number) => string;
  onClose: () => void;
}

export const ErrorDrilldownCard = ({ callType, buckets, valueFormatter, onClose }: ErrorDrilldownCardProps) => (
  <Card className="mt-4">
    <CardHeader className="flex flex-row items-center justify-between">
      <CardTitle className="text-base font-semibold">Failed requests by error code: {callType}</CardTitle>
      <Button variant="outline" size="icon-sm" onClick={onClose} aria-label="Close error breakdown">
        <X />
      </Button>
    </CardHeader>
    <CardContent>
      <p className="text-sm text-muted-foreground">Hover a bar to see the error classes behind that code.</p>
      <BarChart
        data={groupErrorBuckets(buckets, callType)}
        index="error_code"
        categories={[FAILED_REQUESTS_SERIES]}
        colors={["red"]}
        valueFormatter={valueFormatter}
        showLegend={false}
        customTooltip={ErrorCodeTooltip}
        yAxisWidth={48}
        className="mt-2"
      />
    </CardContent>
  </Card>
);
