import React from "react";
import type { ColumnDef } from "@tanstack/react-table";

import { DonutChart } from "@/components/shared/charts";
import { DataTable } from "@/components/shared/DataTable";
import { ChartLoader } from "@/components/shared/chart_loader";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatNumberWithCommas } from "@/utils/dataUtils";

interface SpendByCategoryPanelProps<TRow extends Record<string, unknown>> {
  title: string;
  headerAction?: React.ReactNode;
  loading: boolean;
  isDateChanging: boolean;
  data: TRow[];
  indexKey: keyof TRow & string;
  columns: ColumnDef<TRow>[];
  getRowId: (row: TRow) => string;
  noDataMessage: string;
}

export function SpendByCategoryPanel<TRow extends Record<string, unknown>>({
  title,
  headerAction,
  loading,
  isDateChanging,
  data,
  indexKey,
  columns,
  getRowId,
  noDataMessage,
}: SpendByCategoryPanelProps<TRow>) {
  return (
    <Card className="h-full">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {headerAction && <CardAction className="flex items-center gap-4">{headerAction}</CardAction>}
      </CardHeader>
      <CardContent>
        {loading ? (
          <ChartLoader isDateChanging={isDateChanging} />
        ) : (
          <div className="grid grid-cols-2">
            <DonutChart
              className="mt-4 h-40"
              data={data}
              index={indexKey}
              category="spend"
              valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
              colors={["cyan"]}
              showLabel
              startAngle={90}
              endAngle={-270}
            />
            <DataTable columns={columns} data={data} getRowId={getRowId} noDataMessage={noDataMessage} size="compact" />
          </div>
        )}
      </CardContent>
    </Card>
  );
}

export default SpendByCategoryPanel;
