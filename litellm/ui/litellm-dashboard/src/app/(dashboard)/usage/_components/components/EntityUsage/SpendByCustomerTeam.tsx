import { DataTable } from "@/components/shared/DataTable";
import { MoneyCell } from "@/components/shared/table_cells";
import type { ColumnDef } from "@tanstack/react-table";
import { useTranslation } from "react-i18next";
import i18n from "@/i18n/i18n";
import type { DailyActivityAggregatedResponse } from "@/components/UsagePage/dailyActivityApi";
import { customerTeamSpend, type CustomerTeamSpend } from "./customerTeamSpend";

const columns: ColumnDef<CustomerTeamSpend>[] = [
  {
    header: () => i18n.t("usage.teamUserSpend.columns.team"),
    accessorKey: "teamAlias",
    cell: ({ row }) => (
      <div>
        <div>{row.original.teamAlias}</div>
        {row.original.teamAlias !== row.original.teamId && (
          <div className="text-xs text-muted-foreground">{row.original.teamId}</div>
        )}
      </div>
    ),
  },
  {
    header: () => i18n.t("usage.teamUserSpend.columns.spend"),
    accessorKey: "spend",
    cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={6} />,
  },
  { header: () => i18n.t("usage.teamUserSpend.columns.requests"), accessorKey: "requests" },
  { header: () => i18n.t("usage.teamUserSpend.columns.tokens"), accessorKey: "tokens" },
];

export default function SpendByCustomerTeam({ response }: { response: DailyActivityAggregatedResponse }) {
  const { t } = useTranslation();
  return (
    <section className="grid gap-3">
      <h3 className="font-semibold">{t("usage.customerTeamSpend.title")}</h3>
      <p className="text-xs text-muted-foreground">{t("usage.customerTeamSpend.attributionNote")}</p>
      <DataTable
        columns={columns}
        data={customerTeamSpend(response)}
        getRowId={(row) => row.teamId}
        noDataMessage={t("usage.customerTeamSpend.noData")}
        size="compact"
      />
    </section>
  );
}
