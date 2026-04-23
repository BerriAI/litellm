import React from "react";
import { Progress } from "@/components/ui/progress";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { cn } from "@/lib/utils";
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

const calculateSuccessRate = (successful: number, total: number): number => {
  if (total === 0) return 0;
  return (successful / total) * 100;
};

const EndpointUsageTable: React.FC<EndpointUsageTableProps> = ({
  endpointData,
}) => {
  const rows: EndpointRow[] = Object.entries(endpointData).map(
    ([endpoint, data]) => ({
      key: endpoint,
      endpoint,
      successful_requests: data.metrics.successful_requests,
      failed_requests: data.metrics.failed_requests,
      api_requests: data.metrics.api_requests,
      total_tokens: data.metrics.total_tokens,
      spend: data.metrics.spend,
      successRate: calculateSuccessRate(
        data.metrics.successful_requests,
        data.metrics.api_requests,
      ),
    }),
  );

  return (
    <div className="border border-border rounded-md overflow-hidden">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Endpoint</TableHead>
            <TableHead>Successful / Failed</TableHead>
            <TableHead>Total Request</TableHead>
            <TableHead>Success Rate</TableHead>
            <TableHead>Total Tokens</TableHead>
            <TableHead>Spend</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((record) => {
            const successPercentage =
              record.api_requests > 0
                ? (record.successful_requests / record.api_requests) * 100
                : 0;
            const successRateStr = record.successRate.toFixed(2);
            return (
              <TableRow key={record.key}>
                <TableCell className="font-medium">
                  {record.endpoint}
                </TableCell>
                <TableCell>
                  <div className="flex items-center space-x-3">
                    <div className="flex-1 relative">
                      <Progress value={successPercentage} className="h-2" />
                    </div>
                    <div className="flex items-center space-x-2 text-sm min-w-[100px]">
                      <span className="text-emerald-600 dark:text-emerald-400 font-medium">
                        {record.successful_requests.toLocaleString()}
                      </span>
                      <span className="text-muted-foreground">/</span>
                      <span className="text-destructive font-medium">
                        {record.failed_requests.toLocaleString()}
                      </span>
                    </div>
                  </div>
                </TableCell>
                <TableCell>
                  {record.api_requests.toLocaleString()}
                </TableCell>
                <TableCell>
                  <span
                    className={cn(
                      "font-medium",
                      record.successRate >= 95
                        ? "text-emerald-600 dark:text-emerald-400"
                        : record.successRate >= 80
                          ? "text-amber-600 dark:text-amber-400"
                          : "text-destructive",
                    )}
                  >
                    {successRateStr}%
                  </span>
                </TableCell>
                <TableCell>
                  {record.total_tokens.toLocaleString()}
                </TableCell>
                <TableCell>
                  ${formatNumberWithCommas(record.spend, 2)}
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
};

export default EndpointUsageTable;
