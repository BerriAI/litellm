import type { ColumnDef } from "@tanstack/react-table";
import { Info } from "lucide-react";
import React from "react";
import { Trans, useTranslation } from "react-i18next";
import { ProviderLogo } from "@/components/molecules/models/ProviderLogo";
import { ChartLoader } from "@/components/shared/chart_loader";
import { DataTable } from "@/components/shared/DataTable";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import i18n from "@/i18n/i18n";
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
  const { t } = useTranslation();
  if (account.account_id === null) {
    return <span className="text-muted-foreground">{t("usage.subscriptionCost.card.accountNotSignedIn")}</span>;
  }
  return (
    <div className="flex flex-col">
      <span className="font-mono text-xs break-all whitespace-normal">{account.account_id}</span>
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
  const { t } = useTranslation();
  if (account.account_id === null) return null;
  if (account.fee === null) {
    return (
      <Button variant="outline" size="xs" onClick={() => onSetFee(account)}>
        {t("usage.subscriptionCost.card.setFee")}
      </Button>
    );
  }
  return (
    <div className="flex justify-end gap-2">
      <Button variant="outline" size="xs" onClick={() => onEditFee(account)}>
        {t("usage.subscriptionCost.card.edit")}
      </Button>
      <AlertDialog>
        <AlertDialogTrigger render={<Button variant="outline" size="xs" />}>
          {t("usage.subscriptionCost.card.remove")}
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{t("usage.subscriptionCost.removeDialog.title")}</AlertDialogTitle>
            <AlertDialogDescription>
              <Trans
                i18nKey="usage.subscriptionCost.removeDialog.description"
                components={{ account: <>{account.account_id}</> }}
              />
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>{t("usage.subscriptionCost.removeDialog.cancel")}</AlertDialogCancel>
            <AlertDialogAction variant="destructive" onClick={() => onRemoveFee(account)}>
              {t("usage.subscriptionCost.removeDialog.confirm")}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
};

const baseColumns: ColumnDef<SubscriptionAccountUsage>[] = [
  {
    header: () => i18n.t("usage.subscriptionCost.columns.provider"),
    accessorKey: "custom_llm_provider",
    cell: ({ row }) => (
      <div className="flex items-center gap-2 whitespace-nowrap">
        <ProviderLogo provider={row.original.custom_llm_provider} className="size-4 shrink-0" />
        <span>{row.original.custom_llm_provider}</span>
      </div>
    ),
  },
  {
    header: () => i18n.t("usage.subscriptionCost.columns.account"),
    accessorKey: "account_id",
    cell: ({ row }) => <AccountCell account={row.original} />,
  },
  {
    header: () => i18n.t("usage.subscriptionCost.columns.deployments"),
    id: "deployments",
    cell: ({ row }) =>
      row.original.deployments.length === 0
        ? i18n.t("usage.subscriptionCost.card.noDeployments")
        : row.original.deployments.join(", "),
  },
  {
    header: () => i18n.t("usage.subscriptionCost.columns.monthlyFee"),
    id: "monthly_fee",
    meta: { numeric: true },
    cell: ({ row }) =>
      row.original.fee === null ? (
        <Badge variant="outline">{i18n.t("usage.subscriptionCost.card.notTracked")}</Badge>
      ) : (
        formatFee(row.original.fee.monthly_fee, row.original.fee.currency)
      ),
  },
  {
    header: () => i18n.t("usage.subscriptionCost.columns.billingPeriods"),
    id: "billing_periods",
    cell: ({ row }) =>
      row.original.billing_periods.length === 0 ? (
        <span className="text-muted-foreground">{i18n.t("usage.subscriptionCost.card.noPeriodsInRange")}</span>
      ) : (
        <div className="flex flex-col">
          {row.original.billing_periods.map((period) => (
            <span key={period.start}>{formatBillingPeriod(period)}</span>
          ))}
        </div>
      ),
  },
  {
    header: () => i18n.t("usage.subscriptionCost.columns.fixedCost"),
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
  const { t } = useTranslation();
  if (!failed && (usage === null || usage.accounts.length === 0)) return null;
  const columns = canEdit ? [...baseColumns, actionsColumn({ onSetFee, onEditFee, onRemoveFee })] : baseColumns;
  const accounts = usage?.accounts ?? [];
  const totals = fixedCostByCurrency(accounts);

  return (
    <Card className="col-span-2">
      <CardHeader>
        <CardTitle className="flex items-center gap-1">
          {t("usage.subscriptionCost.card.title")}
          <Tooltip>
            <TooltipTrigger render={<Info className="size-4 text-muted-foreground hover:text-foreground" />} />
            <TooltipContent>{t(ATTRIBUTION_RULE)}</TooltipContent>
          </Tooltip>
        </CardTitle>
      </CardHeader>
      <CardContent>
        {failed && <p className="text-sm text-destructive">{t("usage.subscriptionCost.card.loadFailed")}</p>}
        {!failed && loading && <ChartLoader isDateChanging={isDateChanging} />}
        {!failed && !loading && (
          <div className="flex flex-col gap-4">
            <DataTable
              columns={columns}
              data={accounts}
              getRowId={accountRowId}
              noDataMessage={t("usage.subscriptionCost.card.noAccounts")}
              size="compact"
            />
            <div className="flex flex-wrap items-center gap-x-6 gap-y-1 text-sm">
              {totals.length === 0 ? (
                <span data-testid="fixed-cost-total">{t("usage.subscriptionCost.card.fixedCostNone")}</span>
              ) : (
                totals.map(({ currency, total }) => (
                  <span key={currency} data-testid="fixed-cost-total">
                    {t("usage.subscriptionCost.card.fixedCostTotal", { total: formatFee(total, currency) })}
                  </span>
                ))
              )}
              <span data-testid="metered-spend">
                {t("usage.subscriptionCost.card.meteredSpend", { spend: formatNumberWithCommas(meteredSpend, 2) })}
              </span>
            </div>
            <p className="text-xs text-muted-foreground">{t(ATTRIBUTION_RULE)}</p>
          </div>
        )}
      </CardContent>
    </Card>
  );
};

export default SubscriptionCostCard;
