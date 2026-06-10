import { formatNumberWithCommas } from "@/utils/dataUtils";
import { BarChart, Card, Title } from "@tremor/react";
import { Table } from "antd";
import type { ColumnsType } from "antd/es/table";
import React, { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { TopModelData } from "../types";

interface KeyModelUsageViewProps {
  topModels: TopModelData[];
}

const VISIBLE_ROWS = 5;
// antd Table with size="small" has a row height of ~39px
const ANTD_SMALL_TABLE_ROW_HEIGHT = 39;

const getColumns = (t: TFunction): ColumnsType<TopModelData> => [
  {
    title: t("usagePage.keyModelUsageView.colModel"),
    dataIndex: "model",
    key: "model",
    render: (value) => value || "-",
  },
  {
    title: t("usagePage.keyModelUsageView.colSpendUsd"),
    dataIndex: "spend",
    key: "spend",
    render: (value) => `$${formatNumberWithCommas(value, 2)}`,
  },
  {
    title: t("usagePage.keyModelUsageView.colSuccessful"),
    dataIndex: "successful_requests",
    key: "successful_requests",
    render: (value) => <span className="text-green-600">{value?.toLocaleString() || 0}</span>,
  },
  {
    title: t("usagePage.keyModelUsageView.colFailed"),
    dataIndex: "failed_requests",
    key: "failed_requests",
    render: (value) => <span className="text-red-600">{value?.toLocaleString() || 0}</span>,
  },
  {
    title: t("usagePage.keyModelUsageView.colTokens"),
    dataIndex: "tokens",
    key: "tokens",
    render: (value) => value?.toLocaleString() || 0,
  },
];

const KeyModelUsageView: React.FC<KeyModelUsageViewProps> = ({ topModels }) => {
  const { t } = useTranslation();
  const columns = useMemo(() => getColumns(t), [t]);
  const [viewMode, setViewMode] = useState<"chart" | "table">("table");

  if (topModels.length === 0) {
    return null;
  }

  return (
    <Card className="mt-4">
      <div className="flex justify-between items-center mb-3">
        <Title>{t("usagePage.keyModelUsageView.title")}</Title>
        <div className="flex space-x-2">
          <button
            onClick={() => setViewMode("table")}
            className={`px-3 py-1 text-sm rounded-md ${viewMode === "table" ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-700"}`}
          >
            {t("usagePage.keyModelUsageView.tableView")}
          </button>
          <button
            onClick={() => setViewMode("chart")}
            className={`px-3 py-1 text-sm rounded-md ${viewMode === "chart" ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-700"}`}
          >
            {t("usagePage.keyModelUsageView.chartView")}
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
        <Table
          columns={columns}
          dataSource={topModels}
          rowKey="model"
          size="small"
          pagination={false}
          scroll={topModels.length > VISIBLE_ROWS ? { y: VISIBLE_ROWS * ANTD_SMALL_TABLE_ROW_HEIGHT } : undefined}
        />
      )}
    </Card>
  );
};

export default KeyModelUsageView;
