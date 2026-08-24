import { DonutChart } from "@/components/shared/charts";
import { DataTable } from "@/components/shared/DataTable";
import { MoneyCell } from "@/components/shared/table_cells";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { Info } from "lucide-react";
import type { ColumnDef } from "@tanstack/react-table";
import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
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
    meta: { numeric: true, className: "text-success" },
    cell: ({ row }) => row.original.successful_requests.toLocaleString(),
  },
  {
    header: "Failed",
    accessorKey: "failed_requests",
    meta: { numeric: true, className: "text-destructive" },
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
      <CardHeader>
        <CardTitle>Spend by Provider</CardTitle>
        <CardAction className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <label className="text-sm text-foreground">Show Zero Spend</label>
            <Switch checked={includeZeroSpend} onCheckedChange={setIncludeZeroSpend} />
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-1">
              <label className="text-sm text-foreground">Show Unknown</label>
              <Tooltip>
                <TooltipTrigger render={<Info className="size-4 text-muted-foreground hover:text-foreground" />} />
                <TooltipContent>Requests that failed to route to a provider</TooltipContent>
              </Tooltip>
            </div>
            <Switch checked={includeUnknown} onCheckedChange={setIncludeUnknown} />
          </div>
        </CardAction>
      </CardHeader>
      <CardContent>
        {loading ? (
          <ChartLoader isDateChanging={isDateChanging} />
        ) : (
          <div className="grid grid-cols-2">
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
            <DataTable
              columns={columns}
              data={filteredProviderSpend}
              getRowId={(row) => row.provider}
              noDataMessage="No provider usage data"
              size="compact"
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default SpendByProvider;
