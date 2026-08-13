import { DonutChart } from "@/components/shared/charts";
import { DataTable } from "@/components/shared/DataTable";
import { MoneyCell } from "@/components/shared/table_cells";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { InfoCircleOutlined } from "@ant-design/icons";
import type { ColumnDef } from "@tanstack/react-table";
import { Card, Col, Grid, Switch, Title } from "@tremor/react";
import { Tooltip } from "antd";
import React, { useState } from "react";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { ChartLoader } from "@/components/shared/chart_loader";

type ProviderSpendData = {
  provider: string;
  spend: number;
  requests: number;
  successful_requests: number;
  failed_requests: number;
  tokens: number;
};

interface SpendByProviderProps {
  loading: boolean;
  isDateChanging: boolean;
  providerSpend: ProviderSpendData[];
}

const columns: ColumnDef<ProviderSpendData>[] = [
  {
    header: "Provider",
    accessorKey: "provider",
    cell: ({ row }) => (
      <div className="flex items-center space-x-2">
        {row.original.provider && <ProviderLogo provider={row.original.provider} className="size-4" />}
        <span>{row.original.provider}</span>
      </div>
    ),
  },
  {
    header: "Spend",
    accessorKey: "spend",
    meta: { numeric: true },
    cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={2} />,
  },
  {
    header: "Successful",
    accessorKey: "successful_requests",
    meta: { numeric: true, className: "text-green-600" },
    cell: ({ row }) => row.original.successful_requests.toLocaleString(),
  },
  {
    header: "Failed",
    accessorKey: "failed_requests",
    meta: { numeric: true, className: "text-red-600" },
    cell: ({ row }) => row.original.failed_requests.toLocaleString(),
  },
  {
    header: "Tokens",
    accessorKey: "tokens",
    meta: { numeric: true },
    cell: ({ row }) => row.original.tokens.toLocaleString(),
  },
];

const SpendByProvider: React.FC<SpendByProviderProps> = ({ loading, isDateChanging, providerSpend }) => {
  const [includeZeroSpend, setIncludeZeroSpend] = useState(false);
  const [includeUnknown, setIncludeUnknown] = useState(false);

  const filteredProviderSpend = providerSpend.filter((provider) => {
    const isUnknown = provider.provider?.toLowerCase() === "unknown";

    // If includeUnknown is true, always include unknown provider
    if (isUnknown) {
      return includeUnknown;
    }

    // If includeZeroSpend is true, include all providers (including those with 0 spend)
    // Otherwise, only include providers with spend > 0
    if (includeZeroSpend) {
      return true;
    }

    return provider.spend > 0;
  });

  return (
    <Card className="h-full">
      <div className="flex justify-between items-center mb-4">
        <Title>Spend by Provider</Title>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm text-gray-700">Show Zero Spend</label>
            <Switch checked={includeZeroSpend} onChange={setIncludeZeroSpend} />
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <label className="text-sm text-gray-700">Show Unknown</label>
              <Tooltip title="Requests that failed to route to a provider">
                <InfoCircleOutlined className="text-gray-400 hover:text-gray-600" />
              </Tooltip>
            </div>
            <Switch checked={includeUnknown} onChange={setIncludeUnknown} />
          </div>
        </div>
      </div>
      {loading ? (
        <ChartLoader isDateChanging={isDateChanging} />
      ) : (
        <Grid numItems={2}>
          <Col numColSpan={1}>
            <DonutChart
              className="mt-4 h-40"
              data={filteredProviderSpend}
              index="provider"
              category="spend"
              valueFormatter={(value) => `$${formatNumberWithCommas(value, 2)}`}
              colors={["cyan"]}
              showLabel
              startAngle={90}
              endAngle={-270}
            />
          </Col>
          <Col numColSpan={1}>
            <DataTable
              columns={columns}
              data={filteredProviderSpend}
              getRowId={(row) => row.provider}
              noDataMessage="No provider usage data"
              size="compact"
            />
          </Col>
        </Grid>
      )}
    </Card>
  );
};

export default SpendByProvider;
