import { BarChart } from "@tremor/react";
import { formatNumberWithCommas } from "../utils/dataUtils";
import { useState } from "react";
import { DataTable } from "./view_logs/table";

interface TopModel {
  key: string;
  spend: number;
  successful_requests: number;
  failed_requests: number;
  tokens: number;
}

interface TopModelViewProps {
  topModels: TopModel[];
}

export default function TopModelView({ topModels }: TopModelViewProps) {
  const [modelViewMode, setModelViewMode] = useState<"chart" | "table">("table");

  const columns = [
    {
      header: "Model",
      accessorKey: "key",
      cell: (info: any) => info.getValue() || "-",
    },
    {
      header: "Spend (USD)",
      accessorKey: "spend",
      cell: (info: any) => {
        const value = info.getValue();
        return `$${formatNumberWithCommas(value, 2)}`;
      },
    },
    {
      header: "Successful",
      accessorKey: "successful_requests",
      cell: (info: any) => <span className="text-green-600">{info.getValue()?.toLocaleString() || 0}</span>,
    },
    {
      header: "Failed",
      accessorKey: "failed_requests",
      cell: (info: any) => <span className="text-red-600">{info.getValue()?.toLocaleString() || 0}</span>,
    },
    {
      header: "Tokens",
      accessorKey: "tokens",
      cell: (info: any) => info.getValue()?.toLocaleString() || 0,
    },
  ];
  return (
    <>
      <div className="mb-4 flex justify-end items-center">
        <div className="flex space-x-2">
          <button
            onClick={() => setModelViewMode("table")}
            className={`px-3 py-1 text-sm rounded-md ${modelViewMode === "table" ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-700"}`}
          >
            Table View
          </button>
          <button
            onClick={() => setModelViewMode("chart")}
            className={`px-3 py-1 text-sm rounded-md ${modelViewMode === "chart" ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-700"}`}
          >
            Chart View
          </button>
        </div>
      </div>
      {modelViewMode === "chart" ? (
        <div className="relative">
          <BarChart
            className="mt-4 h-40"
            data={topModels}
            index="key"
            categories={["spend"]}
            colors={["cyan"]}
            valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
            layout="vertical"
            yAxisWidth={200}
            showLegend={false}
          />
        </div>
      ) : (
        <div className="border rounded-lg overflow-auto">
          <DataTable
            columns={columns}
            data={topModels}
            renderSubComponent={() => <></>}
            getRowCanExpand={() => false}
          />
        </div>
      )}
    </>
  );
}
