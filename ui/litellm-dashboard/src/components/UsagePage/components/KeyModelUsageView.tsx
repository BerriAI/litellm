import { formatNumberWithCommas } from "@/utils/dataUtils";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import { BarChart, Card, Title } from "@tremor/react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
import React, { useState } from "react";
import { TopModelData } from "../types";

interface KeyModelUsageViewProps {
  topModels: TopModelData[];
}

const VISIBLE_ROWS = 5;
const SMALL_TABLE_ROW_HEIGHT = 39;

const KeyModelUsageView: React.FC<KeyModelUsageViewProps> = ({ topModels }) => {
  const [viewMode, setViewMode] = useState<"chart" | "table">("table");

  if (topModels.length === 0) {
    return null;
  }

  return (
    <Card className="mt-4">
      <div className="flex justify-between items-center mb-3">
        <Title>Model Usage</Title>
        <div className="flex space-x-2">
          <button
            type="button"
            onClick={() => setViewMode("table")}
            className={cn(
              "px-3 py-1 text-sm rounded-md",
              viewMode === "table"
                ? "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                : "bg-muted text-muted-foreground",
            )}
          >
            Table
          </button>
          <button
            type="button"
            onClick={() => setViewMode("chart")}
            className={cn(
              "px-3 py-1 text-sm rounded-md",
              viewMode === "chart"
                ? "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                : "bg-muted text-muted-foreground",
            )}
          >
            Chart
          </button>
        </div>
      </div>
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
        <div
          className="border border-border rounded-md overflow-hidden"
          style={
            topModels.length > VISIBLE_ROWS
              ? {
                  maxHeight: VISIBLE_ROWS * SMALL_TABLE_ROW_HEIGHT + 40,
                  overflowY: "auto",
                }
              : undefined
          }
        >
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Model</TableHead>
                <TableHead>Spend (USD)</TableHead>
                <TableHead>Successful</TableHead>
                <TableHead>Failed</TableHead>
                <TableHead>Tokens</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {topModels.map((row) => (
                <TableRow key={row.model}>
                  <TableCell>{row.model || "-"}</TableCell>
                  <TableCell>
                    ${formatNumberWithCommas(row.spend, 2)}
                  </TableCell>
                  <TableCell>
                    <span className="text-emerald-600 dark:text-emerald-400">
                      {row.successful_requests?.toLocaleString() || 0}
                    </span>
                  </TableCell>
                  <TableCell>
                    <span className="text-destructive">
                      {row.failed_requests?.toLocaleString() || 0}
                    </span>
                  </TableCell>
                  <TableCell>
                    {row.tokens?.toLocaleString() || 0}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </Card>
  );
};

export default KeyModelUsageView;
