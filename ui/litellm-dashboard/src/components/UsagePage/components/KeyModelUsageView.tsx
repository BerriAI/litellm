import { BarChart } from "@/components/shared/charts";
import { DataTable } from "@/components/shared/DataTable";
import { MoneyCell } from "@/components/shared/table_cells";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import type { ColumnDef } from "@tanstack/react-table";
import React, { useState } from "react";
import { TopModelData } from "../types";

interface KeyModelUsageViewProps {
  topModels: TopModelData[];
}

const VISIBLE_ROWS = 5;
const COMPACT_TABLE_HEADER_HEIGHT = 33;
const COMPACT_TABLE_ROW_HEIGHT = 32;

const columns: ColumnDef<TopModelData>[] = [
  {
    header: "Model",
    accessorKey: "model",
    cell: ({ row }) => row.original.model || "-",
  },
  {
    header: "Spend (USD)",
    accessorKey: "spend",
    meta: { numeric: true },
    cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={2} />,
  },
  {
    header: "Successful",
    accessorKey: "successful_requests",
    meta: { numeric: true },
    cell: ({ row }) => <span className="text-success">{row.original.successful_requests?.toLocaleString() || 0}</span>,
  },
  {
    header: "Failed",
    accessorKey: "failed_requests",
    meta: { numeric: true },
    cell: ({ row }) => <span className="text-destructive">{row.original.failed_requests?.toLocaleString() || 0}</span>,
  },
  {
    header: "Tokens",
    accessorKey: "tokens",
    meta: { numeric: true },
    cell: ({ row }) => row.original.tokens?.toLocaleString() || 0,
  },
];

const KeyModelUsageView: React.FC<KeyModelUsageViewProps> = ({ topModels }) => {
  const [viewMode, setViewMode] = useState<"chart" | "table">("table");

  if (topModels.length === 0) {
    return null;
  }

  return (
    <Card className="mt-4">
      <CardHeader>
        <CardTitle className="text-base font-semibold">Model Usage</CardTitle>
        <CardAction>
          <div className="flex space-x-2">
            <button
              onClick={() => setViewMode("table")}
              className={`px-3 py-1 text-sm rounded-md ${viewMode === "table" ? "bg-info/15 text-info" : "bg-muted text-foreground"}`}
            >
              Table
            </button>
            <button
              onClick={() => setViewMode("chart")}
              className={`px-3 py-1 text-sm rounded-md ${viewMode === "chart" ? "bg-info/15 text-info" : "bg-muted text-foreground"}`}
            >
              Chart
            </button>
          </div>
        </CardAction>
      </CardHeader>
      <CardContent>
        {viewMode === "chart" ? (
          <div className="max-h-[234px] overflow-y-auto">
            <BarChart
              style={{ height: topModels.length * 40 }}
              data={topModels.map((m) => ({ key: m.model, spend: m.spend }))}
              index="key"
              categories={["spend"]}
              colors={["cyan"]}
              valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
              layout="vertical"
              yAxisWidth={180}
              tickGap={5}
              showLegend={false}
            />
          </div>
        ) : (
          <DataTable
            columns={columns}
            data={topModels}
            getRowId={(row) => row.model}
            maxBodyHeight={COMPACT_TABLE_HEADER_HEIGHT + VISIBLE_ROWS * COMPACT_TABLE_ROW_HEIGHT}
            size="compact"
          />
        )}
      </CardContent>
    </Card>
  );
};

export default KeyModelUsageView;
