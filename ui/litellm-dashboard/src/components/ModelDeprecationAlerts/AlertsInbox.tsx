"use client";

import { useMemo, useState } from "react";
import type { ColumnDef } from "@tanstack/react-table";
import { DataTable, DataTableToolbar } from "@/components/shared/DataTable";
import { PageTabs, PageTabsList, PageTabsTrigger, PageTabsContent } from "@/components/shared/Page";
import type { DeprecationAlert } from "./deprecationAlerts";
import { AlertStatus, effectiveAlertStatus } from "./migrationState";
import RecommendedReplacement from "./RecommendedReplacement";
import { DeprecationModel, DeprecationTiming } from "./DeprecationModel";
import { migrationHref } from "./migrationRoute";

const columns: ColumnDef<DeprecationAlert>[] = [
  {
    id: "model",
    header: "Model",
    cell: ({ row }) => (
      <div className="py-1">
        <DeprecationModel alert={row.original} />
        <RecommendedReplacement model={row.original.model.successor_model} />
      </div>
    ),
  },
  {
    id: "deprecation",
    header: "Deprecation",
    cell: ({ row }) => (
      <div className="flex flex-col gap-1">
        <DeprecationTiming alert={row.original} />
        <span className="text-xs text-muted-foreground">{row.original.model.deprecation_date}</span>
      </div>
    ),
  },
  { accessorKey: "requests", header: "Requests · 30d", cell: ({ row }) => row.original.requests.toLocaleString() },
  { accessorKey: "lastUsedDate", header: "Last seen (UTC)" },
  {
    id: "review",
    header: "",
    cell: ({ row }) => (
      <a href={migrationHref(row.original.usageModel)} className="whitespace-nowrap text-primary hover:underline">
        Review impact →
      </a>
    ),
  },
];

export default function AlertsInbox({
  alerts,
  pending,
  status,
  onStatusChange,
  onRefresh,
  refreshing,
}: {
  alerts: readonly DeprecationAlert[];
  pending: boolean;
  status: AlertStatus;
  onStatusChange: (status: AlertStatus) => void;
  onRefresh: () => void;
  refreshing: boolean;
}) {
  const [search, setSearch] = useState("");
  const counts = { open: 0, resolved: 0, dismissed: 0 };
  for (const alert of alerts) counts[effectiveAlertStatus(alert)]++;
  const filtered = useMemo(
    () =>
      alerts.filter(
        (alert) =>
          effectiveAlertStatus(alert) === status && alert.usageModel.toLowerCase().includes(search.toLowerCase()),
      ),
    [alerts, status, search],
  );
  return (
    <PageTabs value={status} onValueChange={onStatusChange} className="gap-4">
      <PageTabsList>
        {Object.values(AlertStatus).map((value) => (
          <PageTabsTrigger key={value} value={value} className="capitalize">
            {value}
            <span className="ml-1.5 text-xs text-muted-foreground">{counts[value]}</span>
          </PageTabsTrigger>
        ))}
      </PageTabsList>
      <PageTabsContent value={status}>
        <DataTable
          data={filtered}
          columns={columns}
          getRowId={(row) => row.id}
          fillHeight
          size="compact"
          isLoading={pending}
          loadingMessage="Loading model deprecations…"
          noDataMessage={`No ${status} model deprecations in the last 30 days.`}
          toolbar={(table) => (
            <DataTableToolbar
              table={table}
              searchValue={search}
              onSearchChange={setSearch}
              searchPlaceholder="Search models…"
              onRefresh={onRefresh}
              isRefreshing={refreshing}
              showViewOptions={false}
            />
          )}
        />
      </PageTabsContent>
    </PageTabs>
  );
}
