import type { ColumnDef } from "@tanstack/react-table";
import { Info } from "lucide-react";
import React from "react";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { ChartLoader } from "@/components/shared/chart_loader";
import { DataTable } from "@/components/shared/DataTable";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { formatNumberWithCommas } from "@/utils/dataUtils";
import { ATTRIBUTION_RULE, fixedCostByCurrency, formatBillingPeriod, formatFee } from "./formatting";
import type { SubscriptionAccountUsage, SubscriptionUsageResponse } from "./types";

interface SubscriptionCostCardProps {
  loading: boolean;
  failed: boolean;
  isDateChanging: boolean;
  usage: SubscriptionUsageResponse | null;
  meteredSpend: number;
  canEdit: boolean;
  onSetFee: (account: SubscriptionAccountUsage) => void;
  onEditFee: (account: SubscriptionAccountUsage) => void;
  onRemoveFee: (account: SubscriptionAccountUsage) => void;
}

type FeeActions = Pick<SubscriptionCostCardProps, "onSetFee" | "onEditFee" | "onRemoveFee">;

const accountRowId = (account: SubscriptionAccountUsage): string =>
  `${account.custom_llm_provider}:${account.account_id ?? "unresolved"}`;

const AccountCell = ({ account }: { account: SubscriptionAccountUsage }) => {
  if (account.account_id === null) {
    return <span className="text-muted-foreground">Account not signed in on the proxy</span>;
  }
  return (
    <div className="flex flex-col">
      <span className="font-mono text-xs">{account.account_id}</span>
      {account.fee?.label && <span className="text-xs text-muted-foreground">{account.fee.label}</span>}
    </div>
  );
};

const ActionsCell = ({
  account,
  onSetFee,
  onEditFee,
  onRemoveFee,
}: { account: SubscriptionAccountUsage } & FeeActions) => {
  if (account.account_id === null) return null;
  if (account.fee === null) {
    return (
      <Button variant="outline" size="xs" onClick={() => onSetFee(account)}>
        Set monthly fee
      </Button>
    );
  }
  return (
    <div className="flex justify-end gap-2">
      <Button variant="outline" size="xs" onClick={() => onEditFee(account)}>
        Edit
      </Button>
      <Button variant="outline" size="xs" onClick={() => onRemoveFee(account)}>
        Remove
      </Button>
    </div>
  );
};

const baseColumns: ColumnDef<SubscriptionAccountUsage>[] = [
  {
    header: "Provider",
    accessorKey: "custom_llm_provider",
    cell: ({ row }) => (
      <div className="flex items-center space-x-2">
        <ProviderLogo provider={row.original.custom_llm_provider} className="size-4" />
        <span>{row.original.custom_llm_provider}</span>
      </div>
    ),
  },
  {
    header: "Account",
    accessorKey: "account_id",
    cell: ({ row }) => <AccountCell account={row.original} />,
  },
  {
    header: "Deployments",
    id: "deployments",
    cell: ({ row }) => (row.original.deployments.length === 0 ? "None" : row.original.deployments.join(", ")),
  },
  {
    header: "Monthly fee",
    id: "monthly_fee",
    meta: { numeric: true },
    cell: ({ row }) =>
      row.original.fee === null ? (
        <Badge variant="outline">Fixed cost not tracked</Badge>
      ) : (
        formatFee(row.original.fee.monthly_fee, row.original.fee.currency)
      ),
  },
  {
    header: "Billing periods in range",
    id: "billing_periods",
    cell: ({ row }) =>
      row.original.billing_periods.length === 0 ? (
        <span className="text-muted-foreground">None in range</span>
      ) : (
        <div className="flex flex-col">
          {row.original.billing_periods.map((period) => (
            <span key={period.start}>{formatBillingPeriod(period)}</span>
          ))}
        </div>
      ),
  },
  {
    header: "Fixed cost in range",
    id: "fixed_cost",
    meta: { numeric: true },
    cell: ({ row }) =>
      row.original.fee === null || row.original.fixed_cost === null
        ? "-"
        : formatFee(row.original.fixed_cost, row.original.fee.currency),
  },
];

const actionsColumn = (actions: FeeActions): ColumnDef<SubscriptionAccountUsage> => ({
  header: "",
  id: "actions",
  meta: { numeric: true },
  cell: ({ row }) => <ActionsCell account={row.original} {...actions} />,
});

const SubscriptionCostCard: React.FC<SubscriptionCostCardProps> = ({
  loading,
  failed,
  isDateChanging,
  usage,
  meteredSpend,
  canEdit,
  onSetFee,
  onEditFee,
  onRemoveFee,
}) => {
  if (!failed && (usage === null || usage.accounts.length === 0)) return null;
  const columns = canEdit ? [...baseColumns, actionsColumn({ onSetFee, onEditFee, onRemoveFee })] : baseColumns;
  const accounts = usage?.accounts ?? [];
  const totals = fixedCostByCurrency(accounts);

  return (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-1">
          Subscription fixed costs
          <Tooltip>
            <TooltipTrigger render={<Info className="size-4 text-muted-foreground hover:text-foreground" />} />
            <TooltipContent>{ATTRIBUTION_RULE}</TooltipContent>
          </Tooltip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {failed && <p className="text-sm text-destructive">Could not load subscription accounts for this range.</p>}
        {!failed && loading && <ChartLoader isDateChanging={isDateChanging} />}
        {!failed && !loading && (
          <div className="flex flex-col gap-4">
            <DataTable
              columns={columns}
              data={accounts}
              getRowId={accountRowId}
              noDataMessage="No subscription accounts"
              size="compact"
            />
            <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-sm">
              {totals.length === 0 ? (
                <span data-testid="fixed-cost-total">Fixed cost in range: none</span>
              ) : (
                totals.map(({ currency, total }) => (
                  <span key={currency} data-testid="fixed-cost-total">
                    Fixed cost in range: {formatFee(total, currency)}
                  </span>
                ))
              )}
              <span data-testid="metered-spend">Metered spend: ${formatNumberWithCommas(meteredSpend, 2)}</span>
            </div>
            <p className="text-xs text-muted-foreground">{ATTRIBUTION_RULE}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default SubscriptionCostCard;
