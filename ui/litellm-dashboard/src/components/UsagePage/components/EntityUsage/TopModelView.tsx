import { BarChart } from "@tremor/react";
import { Segmented } from "antd";
import { useState } from "react";
import { useTranslation } from "react-i18next";
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

export default function TopModelView({ topModels, topModelsLimit, setTopModelsLimit }: TopModelViewProps) {
  const { t } = useTranslation();
  const [modelViewMode, setModelViewMode] = useState<"chart" | "table">("table");

  const columns = [
    {
      header: t("usagePage.topModelView.colModel"),
      accessorKey: "key",
      cell: (info: any) => info.getValue() || "-",
    },
    {
      header: t("usagePage.topModelView.colSpendUsd"),
      accessorKey: "spend",
      cell: (info: any) => {
        const value = info.getValue();
        return `$${formatNumberWithCommas(value, 2)}`;
      },
    },
    {
      header: t("usagePage.topModelView.colSuccessful"),
      accessorKey: "successful_requests",
      cell: (info: any) => <span className="text-green-600">{info.getValue()?.toLocaleString() || 0}</span>,
    },
    {
      header: t("usagePage.topModelView.colFailed"),
      accessorKey: "failed_requests",
      cell: (info: any) => <span className="text-red-600">{info.getValue()?.toLocaleString() || 0}</span>,
    },
    {
      header: t("usagePage.topModelView.colTokens"),
      accessorKey: "tokens",
      cell: (info: any) => info.getValue()?.toLocaleString() || 0,
    },
  ];
  const processedTopModels = topModels.slice(0, topModelsLimit);

  return (
    <>
      <div className="mb-4 flex justify-between items-center">
        <Segmented
          options={[
            { label: "5", value: 5 },
            { label: "10", value: 10 },
            { label: "25", value: 25 },
            { label: "50", value: 50 },
          ]}
          value={topModelsLimit}
          onChange={(value) => setTopModelsLimit(value as number)}
        />
        <div className="flex space-x-2">
          <button
            onClick={() => setModelViewMode("table")}
            className={`px-3 py-1 text-sm rounded-md ${modelViewMode === "table" ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-700"}`}
          >
            {t("usagePage.topModelView.tableView")}
          </button>
          <button
            onClick={() => setModelViewMode("chart")}
            className={`px-3 py-1 text-sm rounded-md ${modelViewMode === "chart" ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-700"}`}
          >
            {t("usagePage.topModelView.chartView")}
          </button>
        </div>
      </div>
      {modelViewMode === "chart" ? (
        <div className="relative max-h-[600px] overflow-y-auto">
          <BarChart
            className="mt-4 cursor-pointer hover:opacity-90"
            style={{ height: Math.min(processedTopModels.length, topModelsLimit) * 52 }}
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
