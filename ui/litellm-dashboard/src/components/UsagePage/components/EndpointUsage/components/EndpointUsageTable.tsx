import React from "react";
import { Table, Progress } from "antd";
import type { ColumnsType } from "antd/es/table";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { MetricWithMetadata } from "../../../types";

interface EndpointUsageTableProps {
  endpointData: Record<string, MetricWithMetadata>;
}

interface EndpointRow {
  key: string;
  endpoint: string;
  successful_requests: number;
  failed_requests: number;
  api_requests: number;
  total_tokens: number;
  spend: number;
  successRate: number;
}

const EndpointUsageTable: React.FC<EndpointUsageTableProps> = ({ endpointData }) => {
  const calculateSuccessRate = (successful: number, total: number): number => {
    if (total === 0) return 0;
    return (successful / total) * 100;
  };

  const dataSource: EndpointRow[] = Object.entries(endpointData).map(([endpoint, data]) => ({
    key: endpoint,
    endpoint,
    successful_requests: data.metrics.successful_requests,
    failed_requests: data.metrics.failed_requests,
    api_requests: data.metrics.api_requests,
    total_tokens: data.metrics.total_tokens,
    spend: data.metrics.spend,
    successRate: calculateSuccessRate(data.metrics.successful_requests, data.metrics.api_requests),
  }));

  const columns: ColumnsType<EndpointRow> = [
    {
      title: "Endpoint",
      dataIndex: "endpoint",
      key: "endpoint",
      render: (text: string) => <span className="font-medium">{text}</span>,
    },
    {
      title: "Successful / Failed",
      key: "requests",
      render: (_: any, record: EndpointRow) => {
        const successPercentage =
          record.api_requests > 0 ? (record.successful_requests / record.api_requests) * 100 : 0;
        const failurePercentage = record.api_requests > 0 ? (record.failed_requests / record.api_requests) * 100 : 0;
        const totalPercentage = successPercentage + failurePercentage;

        const strokeColorConfig: Record<string, string> = {
          "0%": "#22c55e",
        };
        if (successPercentage > 0 && successPercentage < 100) {
          strokeColorConfig[`${successPercentage}%`] = "#22c55e";
          strokeColorConfig[`${successPercentage + 0.01}%`] = "#ef4444";
        }
        strokeColorConfig["100%"] = failurePercentage > 0 ? "#ef4444" : "#22c55e";

        return (
          <div className="flex items-center space-x-3">
            <div className="flex-1 relative">
              <Progress percent={totalPercentage} size="small" strokeColor={strokeColorConfig} showInfo={false} />
            </div>
            <div className="flex items-center space-x-2 text-sm min-w-[100px]">
              <span className="text-green-600 font-medium">{record.successful_requests.toLocaleString()}</span>
              <span className="text-gray-400">/</span>
              <span className="text-red-600 font-medium">{record.failed_requests.toLocaleString()}</span>
            </div>
          </div>
        );
      },
    },
    {
      title: "Total Request",
      dataIndex: "api_requests",
      key: "api_requests",
      render: (value: number) => value.toLocaleString(),
    },
    {
      title: "Success Rate",
      dataIndex: "successRate",
      key: "successRate",
      render: (value: number) => {
        const successRateStr = value.toFixed(2);
        return (
          <span
            className={
              value >= 95
                ? "text-green-600 font-medium"
                : value >= 80
                  ? "text-yellow-600 font-medium"
                  : "text-red-600 font-medium"
            }
          >
            {successRateStr}%
          </span>
        );
      },
    },
    {
      title: "Total Tokens",
      dataIndex: "total_tokens",
      key: "total_tokens",
      render: (value: number) => value.toLocaleString(),
    },
    {
      title: "Spend",
      dataIndex: "spend",
      key: "spend",
      render: (value: number) => `$${formatNumberWithCommas(value, 2)}`,
    },
  ];

  return <Table columns={columns} dataSource={dataSource} pagination={false} />;
};

export default EndpointUsageTable;
