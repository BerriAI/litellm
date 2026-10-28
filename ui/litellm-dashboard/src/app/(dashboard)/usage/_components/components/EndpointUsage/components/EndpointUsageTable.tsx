import React from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { Route } from "lucide-react";
import { cn } from "@/lib/cva.config";
import { Panel } from "../../overview/Primitives";
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
      cell: ({ row }) => <span className="font-mono text-xs text-foreground">{row.original.endpoint}</span>,
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
          <div className="flex items-center gap-3">
            <div className="relative min-w-16 flex-1">
              <Meter value={successPercentage} max={totalPercentage || 100} aria-label="Successful requests">
                <MeterTrack className={failurePercentage > 0 ? "h-1 bg-destructive/70" : "h-1"}>
                  <MeterIndicator className="bg-[#2b3fd6]" />
                </MeterTrack>
              </Meter>
            </div>
            <div className="flex min-w-[100px] items-center gap-1.5 text-xs tabular-nums">
              <span className="text-foreground">{record.successful_requests.toLocaleString()}</span>
              <span className="text-muted-foreground">/</span>
              <span className={record.failed_requests > 0 ? "text-destructive" : "text-muted-foreground"}>
                {record.failed_requests.toLocaleString()}
              </span>
            </div>
          </div>
        );
      },
    },
    {
      header: "Total Request",
      accessorKey: "api_requests",
      meta: { numeric: true },
      cell: ({ row }) => <span className="tabular-nums">{row.original.api_requests.toLocaleString()}</span>,
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
            className={cn(
              "tabular-nums",
              value >= 95 ? "text-foreground" : value >= 80 ? "text-warning" : "text-destructive",
            )}
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
      cell: ({ row }) => <span className="tabular-nums">{row.original.total_tokens.toLocaleString()}</span>,
    },
    {
      header: "Spend",
      accessorKey: "spend",
      meta: { numeric: true },
      cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={2} />,
    },
  ];

  return (
    <Panel
      icon={Route}
      title="Endpoints"
      action={
        <span className="text-xs text-muted-foreground tabular-nums">
          {dataSource.length.toLocaleString()} {dataSource.length === 1 ? "endpoint" : "endpoints"}
        </span>
      }
    >
      <DataTable
        columns={columns}
        data={dataSource}
        getRowId={(row) => row.key}
        noDataMessage="No endpoint usage data"
        size="compact"
      />
    </Panel>
  );
};

export default EndpointUsageTable;
