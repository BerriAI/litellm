import type { ColumnDef } from "@tanstack/react-table";
import type { TFunction } from "i18next";
import { CircleDollarSign } from "lucide-react";
import React, { useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { GuardrailUsageDetail } from "@/app/(dashboard)/hooks/guardrails/useGuardrailsUsage";
import { CalcPopover, MathTable } from "@/components/GuardrailsMonitor/CalcPopover";
import { MetricCard } from "@/components/GuardrailsMonitor/MetricCard";
import { UnpricedNote } from "@/components/GuardrailsMonitor/UnpricedNote";
import {
  counterLabel,
  counterMathRow,
  formatCost,
  totalUnits,
  unitsMathRows,
  unpricedSummary,
} from "@/components/GuardrailsMonitor/usageUnits";
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
  unpriced: number;
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
  untrackedByGroup: GuardrailUsageDetail["untracked_usage_units_by_team"],
): GroupRow[] =>
  Object.entries(unitsByGroup)
    .map(([id, units]) => ({
      id,
      units: totalUnits(units),
      cost: costByGroup[id] ?? null,
      unpriced: totalUnits(untrackedByGroup[id] ?? {}),
    }))
    .sort((a, b) => b.units - a.units);

const UnpricedUnitsCell = ({ unpriced }: { unpriced: number }) =>
  unpriced > 0 ? (
    <span className="text-warning">{unpriced.toLocaleString()}</span>
  ) : (
    <span className="text-muted-foreground">—</span>
  );

const unpricedColumn = <TRow extends { unpriced: number }>(t: TFunction): ColumnDef<TRow> => ({
  header: t("guardrailsMonitor.usageBreakdown.columns.unpricedUnits"),
  accessorKey: "unpriced",
  meta: { numeric: true },
  cell: ({ row }) => <UnpricedUnitsCell unpriced={row.original.unpriced} />,
});

const counterColumns = (t: TFunction): ColumnDef<CounterRow>[] => [
  {
    header: t("guardrailsMonitor.usageBreakdown.columns.counter"),
    accessorKey: "counter",
    cell: ({ row }) => counterLabel(row.original.counter),
  },
  {
    header: t("guardrailsMonitor.usageBreakdown.columns.units"),
    accessorKey: "units",
    meta: { numeric: true },
    cell: ({ row }) => row.original.units.toLocaleString(),
  },
  {
    header: t("guardrailsMonitor.usageBreakdown.columns.cost"),
    accessorKey: "cost",
    meta: { numeric: true },
    cell: ({ row }) => <MoneyCell value={row.original.cost} emptyText="—" showZero />,
  },
  unpricedColumn<CounterRow>(t),
];

const groupColumns = (t: TFunction, label: string, emptyLabel: string): ColumnDef<GroupRow>[] => [
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
    header: t("guardrailsMonitor.usageBreakdown.columns.units"),
    accessorKey: "units",
    meta: { numeric: true },
    cell: ({ row }) => row.original.units.toLocaleString(),
  },
  {
    header: t("guardrailsMonitor.usageBreakdown.columns.cost"),
    accessorKey: "cost",
    meta: { numeric: true },
    cell: ({ row }) => <MoneyCell value={row.original.cost} emptyText="—" showZero />,
  },
  unpricedColumn<GroupRow>(t),
];

const CostMath = ({ counters, detail }: { counters: CounterRow[]; detail: GuardrailUsageDetail }) => {
  const { t } = useTranslation();
  return (
    <CalcPopover
      title={t("guardrailsMonitor.usageBreakdown.costMath.title")}
      formula={t("guardrailsMonitor.usageBreakdown.costMath.formula")}
    >
      <MathTable rows={counters.map((counter) => counterMathRow(counter, t))} total={formatCost(detail.cost)} />
      <p className="text-xs text-muted-foreground">{t("guardrailsMonitor.usageBreakdown.costMath.explanation")}</p>
      <UnpricedNote unpriced={detail.untracked_usage_units} provider={detail.provider} />
    </CalcPopover>
  );
};

const UnitsMath = ({ units }: { units: GuardrailUsageDetail["usage_units"] }) => {
  const { t } = useTranslation();
  return (
    <CalcPopover
      title={t("guardrailsMonitor.usageBreakdown.unitsMath.title")}
      formula={t("guardrailsMonitor.usageBreakdown.unitsMath.formula")}
    >
      <MathTable rows={unitsMathRows(units)} total={totalUnits(units).toLocaleString()} />
      <p className="text-xs text-muted-foreground">{t("guardrailsMonitor.usageBreakdown.unitsMath.explanation")}</p>
    </CalcPopover>
  );
};

const TableHeading = ({ title }: { title: string }) => (
  <h6 className="text-sm font-semibold text-foreground">{title}</h6>
);

export function GuardrailUsageBreakdown({ detail }: { detail: GuardrailUsageDetail }) {
  const { t } = useTranslation();
  const counters = counterRows(detail);
  const unpriced = unpricedSummary(detail.untracked_usage_units, t);
  const counterTableColumns = useMemo(() => counterColumns(t), [t]);
  const teamColumns = useMemo(
    () =>
      groupColumns(t, t("guardrailsMonitor.usageBreakdown.columns.team"), t("guardrailsMonitor.usageBreakdown.noTeam")),
    [t],
  );
  const keyColumns = useMemo(
    () =>
      groupColumns(t, t("guardrailsMonitor.usageBreakdown.columns.key"), t("guardrailsMonitor.usageBreakdown.noKey")),
    [t],
  );

  return (
    <section className="space-y-4" aria-label={t("guardrailsMonitor.usageBreakdown.ariaLabel")}>
      <div>
        <h5 className="mb-0 text-base font-semibold text-foreground">{t("guardrailsMonitor.usageBreakdown.title")}</h5>
        <p className="mt-0.5 text-xs text-muted-foreground">{t("guardrailsMonitor.usageBreakdown.description")}</p>
      </div>

      {counters.length === 0 ? (
        <p className="text-sm text-muted-foreground">{t("guardrailsMonitor.usageBreakdown.empty")}</p>
      ) : (
        <>
          <div className="grid grid-cols-2 gap-4 md:grid-cols-3">
            <MetricCard
              label={t("guardrailsMonitor.usageBreakdown.metrics.cost")}
              value={formatCost(detail.cost)}
              valueColor={detail.cost != null ? "text-foreground" : "text-muted-foreground"}
              icon={<CircleDollarSign className="size-4" />}
              subtitle={unpriced ?? undefined}
              hint={<CostMath counters={counters} detail={detail} />}
            />
            <MetricCard
              label={t("guardrailsMonitor.usageBreakdown.metrics.usageUnits")}
              value={totalUnits(detail.usage_units).toLocaleString()}
              subtitle={t("guardrailsMonitor.usageBreakdown.metrics.counterCount", { count: counters.length })}
              hint={<UnitsMath units={detail.usage_units} />}
            />
          </div>

          <DataTable
            columns={counterTableColumns}
            data={counters}
            getRowId={(row) => row.counter}
            size="compact"
            toolbar={() => <TableHeading title={t("guardrailsMonitor.usageBreakdown.tables.byCounter")} />}
          />

          <div className="grid gap-4 lg:grid-cols-2">
            <DataTable
              columns={teamColumns}
              data={groupRows(detail.usage_units_by_team, detail.cost_by_team, detail.untracked_usage_units_by_team)}
              getRowId={(row) => row.id || "no-team"}
              size="compact"
              toolbar={() => <TableHeading title={t("guardrailsMonitor.usageBreakdown.tables.byTeam")} />}
            />
            <DataTable
              columns={keyColumns}
              data={groupRows(detail.usage_units_by_key, detail.cost_by_key, detail.untracked_usage_units_by_key)}
              getRowId={(row) => row.id || "no-key"}
              size="compact"
              toolbar={() => <TableHeading title={t("guardrailsMonitor.usageBreakdown.tables.byKey")} />}
            />
          </div>
        </>
      )}
    </section>
  );
}
