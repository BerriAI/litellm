import { STACKED_USAGE_PALETTE } from "@/components/shared/charts";
import { DataTable } from "@/components/shared/DataTable";
import { MoneyCell } from "@/components/shared/table_cells";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { Info, Server } from "lucide-react";
import type { ColumnDef } from "@tanstack/react-table";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import React, { useId, useState } from "react";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { ChartLoader } from "@/components/shared/chart_loader";
import { Panel } from "../overview/Primitives";

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

const QUIET_HEADER = { headerClassName: "font-normal" };

const columns: ColumnDef<ProviderSpendData>[] = [
  {
    header: "Provider",
    accessorKey: "provider",
    meta: QUIET_HEADER,
    cell: ({ row }) => (
      <div className="flex min-w-0 items-center gap-2">
        {row.original.provider && (
          <span className="inline-flex size-6 shrink-0 items-center justify-center rounded-md border bg-background">
            <ProviderLogo provider={row.original.provider} className="size-3.5 rounded-[3px]" />
          </span>
        )}
        <span className="truncate font-medium text-foreground">{row.original.provider}</span>
      </div>
    ),
  },
  {
    header: "Spend",
    accessorKey: "spend",
    meta: { numeric: true, ...QUIET_HEADER },
    cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={2} />,
  },
  {
    header: "Successful",
    accessorKey: "successful_requests",
    meta: { numeric: true, className: "text-success", ...QUIET_HEADER },
    cell: ({ row }) => row.original.successful_requests.toLocaleString(),
  },
  {
    header: "Failed",
    accessorKey: "failed_requests",
    meta: { numeric: true, className: "text-destructive", ...QUIET_HEADER },
    cell: ({ row }) => row.original.failed_requests.toLocaleString(),
  },
  {
    header: "Tokens",
    accessorKey: "tokens",
    meta: { numeric: true, ...QUIET_HEADER },
    cell: ({ row }) => row.original.tokens.toLocaleString(),
  },
];

/**
 * One stacked bar of each provider's share of spend over the per-provider table. Reads at a
 * glance where a donut does not, and keeps sub-cent providers legible. Shared with the entity
 * usage views.
 */
export function ProviderSpendBreakdown({ rows }: { rows: readonly ProviderSpendData[] }) {
  const sorted = [...rows].sort((a, b) => b.spend - a.spend);
  const total = sorted.reduce((sum, row) => sum + row.spend, 0);
  const shareOf = (spend: number) => (total > 0 ? (spend / total) * 100 : 0);
  const colorOf = (index: number) => STACKED_USAGE_PALETTE[index % STACKED_USAGE_PALETTE.length];
  const visible = sorted.filter((row) => row.spend > 0);
  return (
    <div className="grid gap-4">
      {visible.length > 0 && (
        <div className="grid gap-2.5">
          <div className="flex items-baseline justify-between gap-3">
            <span className="text-xs text-muted-foreground">Total provider spend</span>
            <span data-testid="provider-spend-total" className="text-sm font-semibold tabular-nums">
              ${formatNumberWithCommas(total, 2)}
            </span>
          </div>
          <div className="flex h-2 w-full overflow-hidden rounded-full bg-muted">
            {visible.map((row, index) => (
              <div
                key={row.provider}
                data-testid="provider-share-segment"
                title={`${row.provider || "unknown"} · ${shareOf(row.spend).toFixed(1)}%`}
                className="h-full first:rounded-l-full last:rounded-r-full"
                style={{ width: `${shareOf(row.spend)}%`, backgroundColor: colorOf(index) }}
              />
            ))}
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1">
            {visible.slice(0, 6).map((row, index) => (
              <span key={row.provider} className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <span className="size-2 rounded-[2px]" style={{ backgroundColor: colorOf(index) }} />
                <span className="text-foreground">{row.provider || "unknown"}</span>
                <span className="tabular-nums">{shareOf(row.spend).toFixed(1)}%</span>
              </span>
            ))}
          </div>
        </div>
      )}
      <DataTable
        columns={columns}
        data={sorted}
        getRowId={(row) => row.provider}
        noDataMessage="No provider usage data"
        size="compact"
      />
    </div>
  );
}

function ToggleField({
  label,
  checked,
  onCheckedChange,
  hint,
}: {
  label: string;
  checked: boolean;
  onCheckedChange: (checked: boolean) => void;
  hint?: string;
}) {
  const id = useId();
  return (
    <div className="flex items-center gap-1.5">
      <Switch id={id} size="sm" checked={checked} onCheckedChange={onCheckedChange} />
      <label htmlFor={id} className="text-xs text-muted-foreground">
        {label}
      </label>
      {hint && (
        <Tooltip>
          <TooltipTrigger render={<Info className="size-3.5 text-muted-foreground hover:text-foreground" />} />
          <TooltipContent>{hint}</TooltipContent>
        </Tooltip>
      )}
    </div>
  );
}

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
    <Panel
      icon={Server}
      title="Spend by Provider"
      className="h-full"
      action={
        <div className="flex flex-wrap items-center gap-3">
          <ToggleField label="Show Zero Spend" checked={includeZeroSpend} onCheckedChange={setIncludeZeroSpend} />
          <ToggleField
            label="Show Unknown"
            checked={includeUnknown}
            onCheckedChange={setIncludeUnknown}
            hint="Requests that failed to route to a provider"
          />
        </div>
      }
    >
      {loading ? (
        <ChartLoader isDateChanging={isDateChanging} />
      ) : (
        <ProviderSpendBreakdown rows={filteredProviderSpend} />
      )}
    </Panel>
  );
};

export default SpendByProvider;
