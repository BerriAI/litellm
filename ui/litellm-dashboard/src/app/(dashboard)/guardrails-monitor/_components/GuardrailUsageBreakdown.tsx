import type { ColumnDef } from "@tanstack/react-table";
import { CircleDollarSign } from "lucide-react";
import React from "react";
import type { GuardrailUsageDetail } from "@/app/(dashboard)/hooks/guardrails/useGuardrailsUsage";
import { MetricCard } from "@/components/GuardrailsMonitor/MetricCard";
import { counterLabel, formatCost, totalUnits, unpricedSummary } from "@/components/GuardrailsMonitor/usageUnits";
import { DataTable } from "@/components/shared/DataTable";
import { IdCell } from "@/components/shared/table_cells/id_cell";
import { MoneyCell } from "@/components/shared/table_cells/money_cell";

interface CounterRow {
  counter: string;
  units: number;
  cost: number | null;
  unpriced: number;
}

interface GroupRow {
  id: string;
  units: number;
  cost: number | null;
}

const counterRows = (detail: GuardrailUsageDetail): CounterRow[] =>
  Object.entries(detail.usage_units).map(([counter, units]) => ({
    counter,
    units,
    cost: detail.cost_by_unit[counter] ?? null,
    unpriced: detail.untracked_usage_units[counter] ?? 0,
  }));

const groupRows = (
  unitsByGroup: GuardrailUsageDetail["usage_units_by_team"],
  costByGroup: GuardrailUsageDetail["cost_by_team"],
): GroupRow[] =>
  Object.entries(unitsByGroup)
    .map(([id, units]) => ({ id, units: totalUnits(units), cost: costByGroup[id] ?? null }))
    .sort((a, b) => b.units - a.units);

const counterColumns: ColumnDef<CounterRow>[] = [
  { header: "Counter", accessorKey: "counter", cell: ({ row }) => counterLabel(row.original.counter) },
  {
    header: "Units",
    accessorKey: "units",
    meta: { numeric: true },
    cell: ({ row }) => row.original.units.toLocaleString(),
  },
  {
    header: "Cost",
    accessorKey: "cost",
    meta: { numeric: true },
    cell: ({ row }) => <MoneyCell value={row.original.cost} emptyText="—" showZero />,
  },
  {
    header: "Unpriced Units",
    accessorKey: "unpriced",
    meta: { numeric: true },
    cell: ({ row }) =>
      row.original.unpriced > 0 ? (
        <span className="text-warning">{row.original.unpriced.toLocaleString()}</span>
      ) : (
        <span className="text-muted-foreground">—</span>
      ),
  },
];

const groupColumns = (label: string, emptyLabel: string): ColumnDef<GroupRow>[] => [
  {
    header: label,
    accessorKey: "id",
    cell: ({ row }) =>
      row.original.id ? (
        <IdCell value={row.original.id} variant="plain" copyable />
      ) : (
        <span className="text-muted-foreground">{emptyLabel}</span>
      ),
  },
  {
    header: "Units",
    accessorKey: "units",
    meta: { numeric: true },
    cell: ({ row }) => row.original.units.toLocaleString(),
  },
  {
    header: "Cost",
    accessorKey: "cost",
    meta: { numeric: true },
    cell: ({ row }) => <MoneyCell value={row.original.cost} emptyText="—" showZero />,
  },
];

const teamColumns = groupColumns("Team", "No team");
const keyColumns = groupColumns("Key", "No key");

const TableHeading = ({ title }: { title: string }) => (
  <h6 className="text-sm font-semibold text-foreground">{title}</h6>
);

export function GuardrailUsageBreakdown({ detail }: { detail: GuardrailUsageDetail }) {
  const counters = counterRows(detail);
  const unpriced = unpricedSummary(detail.untracked_usage_units);

  return (
    <section className="space-y-4" aria-label="Usage and cost">
      <div>
        <h5 className="mb-0 text-base font-semibold text-foreground">Usage &amp; Cost</h5>
        <p className="mt-0.5 text-xs text-muted-foreground">
          Billable units the provider reported for this guardrail and what LiteLLM priced them at
        </p>
      </div>

      {counters.length === 0 ? (
        <p className="text-sm text-muted-foreground">No billable usage units were recorded in this period.</p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
            <MetricCard
              label="Cost"
              value={formatCost(detail.cost)}
              valueColor={detail.cost != null ? "text-foreground" : "text-muted-foreground"}
              icon={<CircleDollarSign className="size-4" />}
              subtitle={unpriced ?? undefined}
            />
            <MetricCard
              label="Usage Units"
              value={totalUnits(detail.usage_units).toLocaleString()}
              subtitle={`${counters.length} ${counters.length === 1 ? "counter" : "counters"}`}
            />
          </div>

          <DataTable
            columns={counterColumns}
            data={counters}
            getRowId={(row) => row.counter}
            size="compact"
            toolbar={() => <TableHeading title="By counter" />}
          />

          <div className="grid gap-4 lg:grid-cols-2">
            <DataTable
              columns={teamColumns}
              data={groupRows(detail.usage_units_by_team, detail.cost_by_team)}
              getRowId={(row) => row.id || "no-team"}
              size="compact"
              toolbar={() => <TableHeading title="By team" />}
            />
            <DataTable
              columns={keyColumns}
              data={groupRows(detail.usage_units_by_key, detail.cost_by_key)}
              getRowId={(row) => row.id || "no-key"}
              size="compact"
              toolbar={() => <TableHeading title="By key" />}
            />
          </div>
        </>
      )}
    </section>
  );
}
