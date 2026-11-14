"use client";

import { useMemo, useState } from "react";
import { useTranslation } from "react-i18next";
import type { ColumnDef } from "@tanstack/react-table";
import { DataTable, DataTableToolbar } from "@/components/shared/DataTable";
import { PageTabs, PageTabsList, PageTabsTrigger, PageTabsContent } from "@/components/shared/Page";
import i18n from "@/i18n/i18n";
import type { DeprecationAlert } from "./deprecationAlerts";
import { ALERT_STATUS_LABEL_KEYS, AlertStatus, effectiveAlertStatus } from "./migrationState";
import RecommendedReplacement from "./RecommendedReplacement";
import { DeprecationModel, DeprecationTiming } from "./DeprecationModel";
import { migrationHref } from "./migrationRoute";

const EMPTY_MESSAGE_KEYS = {
  [AlertStatus.Open]: "alerts.inbox.empty.open",
  [AlertStatus.Resolved]: "alerts.inbox.empty.resolved",
  [AlertStatus.Dismissed]: "alerts.inbox.empty.dismissed",
} as const;

const columns: ColumnDef<DeprecationAlert>[] = [
  {
    id: "model",
    header: () => i18n.t("alerts.inbox.columns.model"),
    cell: ({ row }) => (
      <div className="py-1">
        <DeprecationModel alert={row.original} />
        <RecommendedReplacement recommendation={row.original.model} />
      </div>
    ),
  },
  {
    id: "deprecation",
    header: () => i18n.t("alerts.inbox.columns.deprecation"),
    cell: ({ row }) => (
      <div className="flex flex-col gap-1">
        <DeprecationTiming alert={row.original} />
        <span className="text-xs text-muted-foreground">{row.original.model.deprecation_date}</span>
      </div>
    ),
  },
  {
    accessorKey: "requests",
    header: () => i18n.t("alerts.inbox.columns.requests"),
    cell: ({ row }) => row.original.requests.toLocaleString(),
  },
  { accessorKey: "lastUsedDate", header: () => i18n.t("alerts.inbox.columns.lastSeen") },
  {
    id: "review",
    header: "",
    cell: ({ row }) => (
      <a href={migrationHref(row.original.usageModel)} className="whitespace-nowrap text-primary hover:underline">
        {i18n.t("alerts.inbox.reviewImpact")}
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
  const { t } = useTranslation();
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
            {t(ALERT_STATUS_LABEL_KEYS[value])}
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
          loadingMessage={t("alerts.inbox.loading")}
          noDataMessage={t(EMPTY_MESSAGE_KEYS[status])}
          toolbar={(table) => (
            <DataTableToolbar
              table={table}
              searchValue={search}
              onSearchChange={setSearch}
              searchPlaceholder={t("alerts.inbox.searchPlaceholder")}
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
