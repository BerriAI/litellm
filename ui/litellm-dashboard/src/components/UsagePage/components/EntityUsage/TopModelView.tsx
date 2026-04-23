// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import { BarChart } from "@tremor/react";
import {
  ToggleGroup,
  ToggleGroupItem,
} from "@/components/ui/toggle-group";
import { useState } from "react";
import { cn } from "@/lib/utils";
import { formatNumberWithCommas } from "../../../../utils/dataUtils";
import { DataTable } from "../../../view_logs/table";

interface TopModel {
  key: string;
  spend: number;
  successful_requests: number;
  failed_requests: number;
  tokens: number;
}

interface TopModelViewProps {
  topModels: TopModel[];
  topModelsLimit: number;
  setTopModelsLimit: (limit: number) => void;
}

export default function TopModelView({
  topModels,
  topModelsLimit,
  setTopModelsLimit,
}: TopModelViewProps) {
  const [modelViewMode, setModelViewMode] = useState<"chart" | "table">(
    "table",
  );

  const columns = [
    {
      header: "Model",
      accessorKey: "key",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      cell: (info: any) => info.getValue() || "-",
    },
    {
      header: "Spend (USD)",
      accessorKey: "spend",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      cell: (info: any) => {
        const value = info.getValue();
        return `$${formatNumberWithCommas(value, 2)}`;
      },
    },
    {
      header: "Successful",
      accessorKey: "successful_requests",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      cell: (info: any) => (
        <span className="text-emerald-600 dark:text-emerald-400">
          {info.getValue()?.toLocaleString() || 0}
        </span>
      ),
    },
    {
      header: "Failed",
      accessorKey: "failed_requests",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      cell: (info: any) => (
        <span className="text-destructive">
          {info.getValue()?.toLocaleString() || 0}
        </span>
      ),
    },
    {
      header: "Tokens",
      accessorKey: "tokens",
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      cell: (info: any) => info.getValue()?.toLocaleString() || 0,
    },
  ];
  const processedTopModels = topModels.slice(0, topModelsLimit);

  return (
    <>
      <div className="mb-4 flex justify-between items-center">
        <ToggleGroup
          type="single"
          value={String(topModelsLimit)}
          onValueChange={(v) => {
            if (!v) return;
            setTopModelsLimit(parseInt(v));
          }}
        >
          {[5, 10, 25, 50].map((n) => (
            <ToggleGroupItem key={n} value={String(n)}>
              {n}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
        <div className="flex space-x-2">
          <button
            type="button"
            onClick={() => setModelViewMode("table")}
            className={cn(
              "px-3 py-1 text-sm rounded-md",
              modelViewMode === "table"
                ? "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                : "bg-muted text-muted-foreground",
            )}
          >
            Table View
          </button>
          <button
            type="button"
            onClick={() => setModelViewMode("chart")}
            className={cn(
              "px-3 py-1 text-sm rounded-md",
              modelViewMode === "chart"
                ? "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300"
                : "bg-muted text-muted-foreground",
            )}
          >
            Chart View
          </button>
        </div>
      </div>
      {modelViewMode === "chart" ? (
        <div className="relative max-h-[600px] overflow-y-auto">
          <BarChart
            className="mt-4 cursor-pointer hover:opacity-90"
            style={{
              height:
                Math.min(processedTopModels.length, topModelsLimit) * 52,
            }}
            data={processedTopModels}
            index="key"
            categories={["spend"]}
            colors={["cyan"]}
            valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
            layout="vertical"
            yAxisWidth={200}
            tickGap={5}
            showLegend={false}
          />
        </div>
      ) : (
        <div className="border rounded-lg overflow-hidden max-h-[600px] overflow-y-auto">
          <DataTable
            columns={columns}
            data={processedTopModels}
            renderSubComponent={() => <></>}
            getRowCanExpand={() => false}
            isLoading={false}
          />
        </div>
      )}
    </>
  );
}
