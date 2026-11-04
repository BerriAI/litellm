"use client";

import { ColumnDef } from "@tanstack/react-table";
import type { ParseKeys, TFunction } from "i18next";

import { DateCell, IdCell, IdentityCell, StatusBadge, type StatusTone } from "@/components/shared/table_cells";

import DefaultProxyAdminTag from "../../common_components/DefaultProxyAdminTag";

export type AuditLogEntry = {
  id: string;
  updated_at: string;
  changed_by: string;
  changed_by_api_key: string;
  action: string;
  table_name: string;
  object_id: string;
  before_value: Record<string, unknown>;
  updated_values: Record<string, unknown>;
};

export const AUDIT_TABLE_NAME_DISPLAY: Record<string, ParseKeys> = {
  LiteLLM_VerificationToken: "logs.audit.tableNames.keys",
  LiteLLM_TeamTable: "logs.audit.tableNames.teams",
  LiteLLM_UserTable: "logs.audit.tableNames.users",
  LiteLLM_OrganizationTable: "logs.audit.tableNames.organizations",
  LiteLLM_ProxyModelTable: "logs.audit.tableNames.models",
  LiteLLM_AgentsTable: "logs.audit.tableNames.agents",
};

const ACTION_TONE: Record<string, StatusTone> = {
  created: "success",
  updated: "info",
  deleted: "error",
  rotated: "warning",
  kill_switch_fired: "error",
};

export const auditActionLabel = (action: string): string => {
  const words = action.replace(/_/g, " ");
  return words ? words.charAt(0).toUpperCase() + words.slice(1) : words;
};

interface AuditLogsTableColumnsDeps {
  onViewLog: (log: AuditLogEntry) => void;
}

export const getAuditLogsTableColumns = (
  { onViewLog }: AuditLogsTableColumnsDeps,
  t: TFunction,
): ColumnDef<AuditLogEntry>[] => [
  {
    id: "updated_at",
    accessorKey: "updated_at",
    header: t("logs.audit.columns.timestamp"),
    size: 200,
    enableSorting: false,
    cell: ({ row }) => <DateCell value={row.original.updated_at} />,
  },
  {
    id: "action",
    accessorKey: "action",
    header: t("logs.audit.columns.action"),
    size: 110,
    enableSorting: false,
    cell: ({ row }) => (
      <StatusBadge tone={ACTION_TONE[row.original.action] ?? "neutral"} label={auditActionLabel(row.original.action)} />
    ),
  },
  {
    id: "table_name",
    accessorKey: "table_name",
    header: t("logs.audit.columns.table"),
    size: 130,
    enableSorting: false,
    cell: ({ row }) => (
      <span className="text-sm">
        {Object.hasOwn(AUDIT_TABLE_NAME_DISPLAY, row.original.table_name)
          ? t(AUDIT_TABLE_NAME_DISPLAY[row.original.table_name])
          : row.original.table_name}
      </span>
    ),
  },
  {
    id: "object_id",
    accessorKey: "object_id",
    header: t("logs.audit.columns.objectId"),
    minSize: 220,
    enableSorting: false,
    cell: ({ row }) => (
      <IdentityCell
        title={row.original.object_id}
        titleClassName="font-mono text-xs font-normal text-primary"
        className="max-w-72"
        onClick={() => onViewLog(row.original)}
      />
    ),
  },
  {
    id: "changed_by",
    accessorKey: "changed_by",
    header: t("logs.audit.columns.changedBy"),
    size: 200,
    enableSorting: false,
    cell: ({ row }) => <DefaultProxyAdminTag userId={row.original.changed_by} />,
  },
  {
    id: "changed_by_api_key",
    accessorKey: "changed_by_api_key",
    header: t("logs.audit.columns.apiKeyHash"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => <IdCell value={row.original.changed_by_api_key} variant="plain" />,
  },
];
