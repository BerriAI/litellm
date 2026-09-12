import React from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { Meter, MeterIndicator, MeterTrack } from "@/components/shared/Meter";
import { DataTable } from "@/components/shared/DataTable";
import { MoneyCell } from "@/components/shared/table_cells";
import { MetricWithMetadata } from "@/components/UsagePage/types";

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

  const columns: ColumnDef<EndpointRow>[] = [
    {
      header: "Endpoint",
      accessorKey: "endpoint",
      cell: ({ row }) => <span className="font-medium">{row.original.endpoint}</span>,
    },
    {
      header: "Successful / Failed",
      id: "requests",
      cell: ({ row }) => {
        const record = row.original;
        const successPercentage =
          record.api_requests > 0 ? (record.successful_requests / record.api_requests) * 100 : 0;
        const failurePercentage = record.api_requests > 0 ? (record.failed_requests / record.api_requests) * 100 : 0;
        const totalPercentage = successPercentage + failurePercentage;

        return (
          <div className="flex items-center space-x-3">
            <div className="flex-1 relative">
              <Meter value={successPercentage} max={totalPercentage || 100} aria-label="Successful requests">
                <MeterTrack className={failurePercentage > 0 ? "bg-destructive" : undefined}>
                  <MeterIndicator className="bg-success" />
                </MeterTrack>
              </Meter>
            </div>
            <div className="flex items-center space-x-2 text-sm min-w-[100px]">
              <span className="text-success font-medium">{record.successful_requests.toLocaleString()}</span>
              <span className="text-muted-foreground">/</span>
              <span className="text-destructive font-medium">{record.failed_requests.toLocaleString()}</span>
            </div>
          </div>
        );
      },
    },
    {
      header: "Total Request",
      accessorKey: "api_requests",
      meta: { numeric: true },
      cell: ({ row }) => row.original.api_requests.toLocaleString(),
    },
    {
      header: "Success Rate",
      accessorKey: "successRate",
      meta: { numeric: true },
      cell: ({ row }) => {
        const value = row.original.successRate;
        const successRateStr = value.toFixed(2);
        return (
          <span
            className={
              value >= 95
                ? "text-success font-medium"
                : value >= 80
                  ? "text-warning font-medium"
                  : "text-destructive font-medium"
            }
          >
            {successRateStr}%
          </span>
        );
      },
    },
    {
      header: "Total Tokens",
      accessorKey: "total_tokens",
      meta: { numeric: true },
      cell: ({ row }) => row.original.total_tokens.toLocaleString(),
    },
    {
      header: "Spend",
      accessorKey: "spend",
      meta: { numeric: true },
      cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={2} />,
    },
  ];

  return (
    <DataTable
      columns={columns}
      data={dataSource}
      getRowId={(row) => row.key}
      noDataMessage="No endpoint usage data"
      size="compact"
    />
  );
};

export default EndpointUsageTable;
