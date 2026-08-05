"use client";

import { ColumnDef } from "@tanstack/react-table";

import { BadgeLink } from "@/components/shared/BadgeLink";
import { DateCell, IdCell, StatusBadge, type StatusTone } from "@/components/shared/table_cells";
import { keyDetailHref, modelDetailHref, orgDetailHref, teamDetailHref, userDetailHref } from "@/utils/entityLinks";

import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";

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
  object_alias?: string | null;
  changed_by_user_email?: string | null;
  changed_by_key_alias?: string | null;
};

export const AUDIT_TABLE_NAME_DISPLAY: Record<string, string> = {
  LiteLLM_VerificationToken: "Keys",
  LiteLLM_TeamTable: "Teams",
  LiteLLM_UserTable: "Users",
  LiteLLM_OrganizationTable: "Organizations",
  LiteLLM_ProxyModelTable: "Models",
};

const OBJECT_HREF_BY_TABLE: Record<string, (objectId: string) => string> = {
  LiteLLM_VerificationToken: keyDetailHref,
  LiteLLM_TeamTable: teamDetailHref,
  LiteLLM_UserTable: userDetailHref,
  LiteLLM_OrganizationTable: orgDetailHref,
  LiteLLM_ProxyModelTable: modelDetailHref,
};

const auditObjectDetailHref = (tableName: string, objectId: string): string | undefined =>
  objectId ? OBJECT_HREF_BY_TABLE[tableName]?.(objectId) : undefined;

const ACTION_TONE: Record<string, StatusTone> = {
  created: "success",
  updated: "info",
  deleted: "error",
  rotated: "warning",
};

const DEFAULT_PROXY_ADMIN_USER_ID = "default_user_id";

const capitalize = (value: string): string => (value ? value.charAt(0).toUpperCase() + value.slice(1) : value);

function ChangedByCell({ userId, userEmail }: { userId: string; userEmail: string | null | undefined }) {
  if (!userId || userId === DEFAULT_PROXY_ADMIN_USER_ID) {
    return <DefaultProxyAdminTag userId={userId} />;
  }

  return (
    <div className="flex min-w-0 flex-col items-start gap-0.5">
      <BadgeLink href={userDetailHref(userId)} className="max-w-full font-normal">
        <span className="truncate">{userEmail || userId}</span>
      </BadgeLink>
      {!!userEmail && (
        <span className="max-w-full truncate font-mono text-xs text-muted-foreground" title={userId}>
          {userId}
        </span>
      )}
    </div>
  );
}

function ChangedByApiKeyCell({ keyHash, keyAlias }: { keyHash: string; keyAlias: string | null | undefined }) {
  if (!keyAlias) {
    return <IdCell value={keyHash} variant="plain" />;
  }

  return (
    <div className="flex min-w-0 flex-col items-start gap-0.5">
      <span className="max-w-full truncate text-sm" title={keyAlias}>
        {keyAlias}
      </span>
      <IdCell value={keyHash} variant="plain" className="text-muted-foreground" />
    </div>
  );
}

export const getAuditLogsTableColumns = (): ColumnDef<AuditLogEntry>[] => [
  {
    id: "updated_at",
    accessorKey: "updated_at",
    header: "Timestamp",
    size: 200,
    enableSorting: false,
    cell: ({ row }) => <DateCell value={row.original.updated_at} />,
  },
  {
    id: "action",
    accessorKey: "action",
    header: "Action",
    size: 110,
    enableSorting: false,
    cell: ({ row }) => (
      <StatusBadge tone={ACTION_TONE[row.original.action] ?? "neutral"} label={capitalize(row.original.action)} />
    ),
  },
  {
    id: "table_name",
    accessorKey: "table_name",
    header: "Table",
    size: 130,
    enableSorting: false,
    cell: ({ row }) => (
      <span className="text-sm">{AUDIT_TABLE_NAME_DISPLAY[row.original.table_name] ?? row.original.table_name}</span>
    ),
  },
  {
    id: "object_id",
    accessorKey: "object_id",
    header: "Object ID",
    minSize: 220,
    enableSorting: false,
    cell: ({ row }) => (
      <BadgeLink
        href={auditObjectDetailHref(row.original.table_name, row.original.object_id)}
        className="max-w-72 font-mono text-xs font-normal"
      >
        <span className="truncate">{row.original.object_id}</span>
      </BadgeLink>
    ),
  },
  {
    id: "object_alias",
    accessorKey: "object_alias",
    header: "Alias",
    size: 160,
    enableSorting: false,
    cell: ({ row }) =>
      row.original.object_alias ? (
        <span className="block max-w-56 truncate text-sm" title={row.original.object_alias}>
          {row.original.object_alias}
        </span>
      ) : (
        <span className="text-muted-foreground">—</span>
      ),
  },
  {
    id: "changed_by",
    accessorKey: "changed_by",
    header: "Changed By",
    size: 200,
    enableSorting: false,
    cell: ({ row }) => (
      <ChangedByCell userId={row.original.changed_by} userEmail={row.original.changed_by_user_email} />
    ),
  },
  {
    id: "changed_by_api_key",
    accessorKey: "changed_by_api_key",
    header: "API Key (Hash)",
    size: 160,
    enableSorting: false,
    cell: ({ row }) => (
      <ChangedByApiKeyCell keyHash={row.original.changed_by_api_key} keyAlias={row.original.changed_by_key_alias} />
    ),
  },
];
