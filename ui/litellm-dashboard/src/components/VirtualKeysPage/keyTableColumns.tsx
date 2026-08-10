"use client";

import { InfoCircleOutlined } from "@ant-design/icons";
import { ColumnDef } from "@tanstack/react-table";
import { Popover, Typography } from "antd";
import type { TFunction } from "i18next";

import { DataTableMultiSortHeader, DataTableSortHeader, type DataTableSortField } from "@/components/shared/DataTable";
import { Skeleton } from "@/components/ui/skeleton";
import {
  DateCell,
  IdCell,
  IdentityCell,
  ModelsCell,
  SpendBudgetCell,
  StatusBadge,
  type StatusTone,
} from "@/components/shared/table_cells";

import DefaultProxyAdminTag from "../common_components/DefaultProxyAdminTag";
import { KeyResponse, Team } from "../key_team_helpers/key_list";
import { Organization } from "../networking";

interface KeyStatus {
  tone: StatusTone;
  label: string;
  tooltip?: string;
}

const getKeyStatus = (key: KeyResponse, t: TFunction<"gateway">): KeyStatus => {
  if (key.blocked === true) {
    const isScimBlocked = (key.metadata as Record<string, unknown> | null | undefined)?.scim_blocked === true;
    return {
      tone: "error",
      label: t("virtualKeys.status.blocked"),
      tooltip: isScimBlocked ? t("virtualKeys.status.scimBlockedTooltip") : t("virtualKeys.status.blockedTooltip"),
    };
  }
  const expiresAt = key.expires ? Date.parse(key.expires) : Number.NaN;
  if (!Number.isNaN(expiresAt) && expiresAt < Date.now()) {
    return {
      tone: "warning",
      label: t("virtualKeys.status.expired"),
      tooltip: t("virtualKeys.status.expiredTooltip"),
    };
  }
  return {
    tone: "success",
    label: t("virtualKeys.status.active"),
    tooltip: t("virtualKeys.status.activeTooltip"),
  };
};

const UserPopoverCell = ({
  userAlias,
  userEmail,
  userId,
  width,
  t,
}: {
  userAlias: string | null;
  userEmail: string | null;
  userId: string | null;
  width: number;
  t: TFunction<"gateway">;
}) => {
  const displayValue = userAlias || userEmail || userId;
  const isDefaultAdmin = userId === "default_user_id";

  const popoverContent = (
    <div className="flex flex-col gap-2 text-xs min-w-[200px] max-w-[300px]">
      {[
        { label: t("virtualKeys.columns.userAlias"), value: userAlias },
        { label: t("virtualKeys.columns.userEmail"), value: userEmail },
        { label: t("virtualKeys.columns.userId"), value: userId },
      ].map(({ label, value }) => (
        <div key={label} className="flex flex-col min-w-0">
          <span className="text-gray-400">{label}</span>
          {value ? (
            <Typography.Text className="font-mono text-xs" ellipsis={{ tooltip: value }} copyable>
              {value}
            </Typography.Text>
          ) : (
            <span className="font-mono">-</span>
          )}
        </div>
      ))}
    </div>
  );

  if (isDefaultAdmin && !userAlias && !userEmail) {
    return (
      <Popover content={popoverContent} trigger="hover" placement="bottomLeft">
        <span className="cursor-default">
          <DefaultProxyAdminTag userId={userId} label={t("virtualKeys.values.defaultProxyAdmin")} />
        </span>
      </Popover>
    );
  }

  return (
    <Popover content={popoverContent} trigger="hover" placement="bottomLeft">
      <span className="font-mono text-xs truncate block cursor-default" style={{ maxWidth: width, overflow: "hidden" }}>
        {displayValue || "-"}
      </span>
    </Popover>
  );
};

const InfoHeader = ({ label, tooltip }: { label: string; tooltip: string }) => (
  <span className="flex items-center gap-1">
    {label}
    <Popover content={tooltip} trigger="hover">
      <InfoCircleOutlined className="text-gray-400 text-xs cursor-help" />
    </Popover>
  </span>
);

interface KeyTableColumnsDeps {
  allTeams: Team[];
  organizations: Organization[];
  onSelectKey: (key: KeyResponse) => void;
  t: TFunction<"gateway">;
  commonT: TFunction<"common">;
  locale: string;
}

export const getKeyTableColumns = ({
  allTeams,
  organizations,
  onSelectKey,
  t,
  commonT,
  locale,
}: KeyTableColumnsDeps): ColumnDef<KeyResponse>[] => {
  const spendBudgetSortFields: DataTableSortField[] = [
    { id: "spend", label: t("virtualKeys.columns.spend") },
    { id: "max_budget", label: t("virtualKeys.columns.budget") },
  ];

  return [
    {
      id: "key_alias",
      accessorKey: "key_alias",
      meta: {
        title: t("virtualKeys.columns.key"),
        renderSkeleton: () => (
          <div className="flex flex-col gap-1 py-1">
            <Skeleton className="h-4 w-32" />
            <div className="flex items-center gap-2">
              <Skeleton className="h-3 w-20" />
              <Skeleton className="h-5 w-16 rounded-full" />
            </div>
          </div>
        ),
      },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={t("virtualKeys.columns.key")} variant="header-cycle" />
      ),
      size: 260,
      enableSorting: true,
      cell: ({ row }) => {
        const status = getKeyStatus(row.original, t);
        return (
          <IdentityCell
            title={row.original.key_alias || "-"}
            subtitle={row.original.key_name}
            badge={
              <StatusBadge
                tone={status.tone}
                label={status.label}
                tooltip={status.tooltip}
                dataTestId={`key-status-${row.original.token_id}`}
              />
            }
            onClick={() => onSelectKey(row.original)}
          />
        );
      },
    },
    {
      id: "token",
      accessorKey: "token",
      meta: { title: t("virtualKeys.columns.keyId") },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={t("virtualKeys.columns.keyId")} variant="header-cycle" />
      ),
      size: 120,
      enableSorting: true,
      cell: (info) => (
        <IdCell value={info.getValue() as string | null} onClick={() => onSelectKey(info.row.original)} />
      ),
    },
    {
      id: "team_alias",
      accessorKey: "team_id",
      meta: { title: t("virtualKeys.columns.team") },
      header: t("virtualKeys.columns.team"),
      size: 120,
      enableSorting: false,
      cell: (info) => {
        const teamId = info.getValue() as string | null;
        if (!teamId) return "-";
        const team = allTeams.find((t) => t.team_id === teamId);
        const displayValue = team?.team_alias || teamId;
        const width = info.cell.column.getSize();
        return (
          <span className="font-mono text-xs truncate block" style={{ maxWidth: width, overflow: "hidden" }}>
            {displayValue}
          </span>
        );
      },
    },
    {
      id: "organization_alias",
      accessorKey: "org_id",
      meta: { title: t("virtualKeys.columns.organization") },
      header: t("virtualKeys.columns.organization"),
      size: 140,
      enableSorting: false,
      cell: (info) => {
        const orgId = info.getValue() as string | null;
        if (!orgId) return "-";
        const org = organizations.find((o) => o.organization_id === orgId);
        const displayValue = org?.organization_alias || orgId;
        const width = info.cell.column.getSize();
        return (
          <span className="font-mono text-xs truncate block" style={{ maxWidth: width, overflow: "hidden" }}>
            {displayValue}
          </span>
        );
      },
    },
    {
      id: "user",
      accessorKey: "user",
      meta: { title: t("virtualKeys.columns.user") },
      header: () => <InfoHeader label={t("virtualKeys.columns.user")} tooltip={t("virtualKeys.columns.userTooltip")} />,
      size: 160,
      enableSorting: false,
      cell: ({ row }) => {
        const key = row.original;
        return (
          <UserPopoverCell
            userAlias={key.user?.user_alias ?? null}
            userEmail={key.user?.user_email ?? key.user_email ?? null}
            userId={key.user_id ?? null}
            width={160}
            t={t}
          />
        );
      },
    },
    {
      id: "created_at",
      accessorKey: "created_at",
      meta: { title: t("virtualKeys.columns.createdAt") },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={t("virtualKeys.columns.createdAt")} variant="header-cycle" />
      ),
      size: 120,
      enableSorting: true,
      cell: (info) => <DateCell value={info.getValue() as string | null} precision="date" locale={locale} />,
    },
    {
      id: "created_by",
      accessorKey: "created_by",
      meta: { title: t("virtualKeys.columns.createdBy") },
      header: t("virtualKeys.columns.createdBy"),
      size: 160,
      enableSorting: false,
      cell: (info) => {
        const userId = info.getValue() as string | null;
        if (!userId) return "-";
        const createdByUser = info.row.original.created_by_user;
        return (
          <UserPopoverCell
            userAlias={createdByUser?.user_alias ?? null}
            userEmail={createdByUser?.user_email ?? null}
            userId={userId}
            width={160}
            t={t}
          />
        );
      },
    },
    {
      id: "updated_at",
      accessorKey: "updated_at",
      meta: { title: t("virtualKeys.columns.updatedAt") },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={t("virtualKeys.columns.updatedAt")} variant="header-cycle" />
      ),
      size: 120,
      enableSorting: true,
      cell: (info) => (
        <DateCell
          value={info.getValue() as string | null}
          precision="date"
          fallback={t("virtualKeys.values.never")}
          locale={locale}
        />
      ),
    },
    {
      id: "last_active",
      accessorKey: "last_active",
      meta: { title: t("virtualKeys.columns.lastActive") },
      header: () => (
        <InfoHeader label={t("virtualKeys.columns.lastActive")} tooltip={t("virtualKeys.columns.lastActiveTooltip")} />
      ),
      size: 190,
      enableSorting: false,
      cell: (info) => (
        <DateCell
          value={info.getValue() as string | null}
          precision="date"
          fallback={t("virtualKeys.values.unknown")}
          locale={locale}
        />
      ),
    },
    {
      id: "expires",
      accessorKey: "expires",
      meta: { title: t("virtualKeys.columns.expires") },
      header: t("virtualKeys.columns.expires"),
      size: 120,
      enableSorting: false,
      cell: (info) => (
        <DateCell
          value={info.getValue() as string | null}
          precision="date"
          fallback={t("virtualKeys.values.never")}
          locale={locale}
        />
      ),
    },
    {
      id: "spend",
      accessorKey: "spend",
      meta: { title: t("virtualKeys.columns.spendBudget"), skeleton: "meter" },
      header: ({ table }) => (
        <DataTableMultiSortHeader
          table={table}
          fields={spendBudgetSortFields}
          labels={{
            ascending: commonT("table.ascending"),
            descending: commonT("table.descending"),
            reset: commonT("table.reset"),
            sortOptions: (fields) => commonT("table.sortOptions", { fields }),
          }}
        />
      ),
      size: 180,
      enableSorting: true,
      cell: ({ row }) => {
        const teamId = row.original.team_id;
        const team = allTeams.find((t) => t.team_id === teamId);
        return (
          <SpendBudgetCell
            spend={row.original.spend}
            maxBudget={row.original.max_budget}
            teamMaxBudget={team?.max_budget ?? null}
            labels={{
              unlimited: t("virtualKeys.values.unlimited"),
              of: t("virtualKeys.values.of"),
              team: t("virtualKeys.values.teamBudget"),
            }}
          />
        );
      },
    },
    {
      id: "budget_reset_at",
      accessorKey: "budget_reset_at",
      meta: { title: t("virtualKeys.columns.budgetReset") },
      header: t("virtualKeys.columns.budgetReset"),
      size: 130,
      enableSorting: false,
      cell: (info) => (
        <DateCell value={info.getValue() as string | null} fallback={t("virtualKeys.values.never")} locale={locale} />
      ),
    },
    {
      id: "models",
      accessorKey: "models",
      meta: { title: t("virtualKeys.columns.models"), skeleton: "chips" },
      header: t("virtualKeys.columns.models"),
      size: 220,
      enableSorting: false,
      cell: (info) => (
        <ModelsCell
          models={info.getValue() as string[] | null | undefined}
          allowedRoutes={info.row.original.allowed_routes}
          keyType={info.row.original.key_type}
          labels={{
            allProxyModels: t("virtualKeys.values.allProxyModels"),
            noModelAccess: t("virtualKeys.values.noModelAccess"),
            scopedRoutes: (scope) => t("virtualKeys.values.scopedRoutes", { scope }),
            more: (count) => t("virtualKeys.values.more", { count }),
          }}
        />
      ),
    },
    {
      id: "rate_limits",
      meta: { title: t("virtualKeys.columns.rateLimits") },
      header: t("virtualKeys.columns.rateLimits"),
      size: 140,
      enableSorting: false,
      cell: ({ row }) => {
        const key = row.original;
        return (
          <div className="text-xs">
            <div>TPM: {key.tpm_limit !== null ? key.tpm_limit : t("virtualKeys.values.unlimited")}</div>
            <div>RPM: {key.rpm_limit !== null ? key.rpm_limit : t("virtualKeys.values.unlimited")}</div>
          </div>
        );
      },
    },
  ];
};

export const KEY_TABLE_HIDDEN_COLUMNS: Record<string, boolean> = {
  token: false,
  organization_alias: false,
  created_by: false,
  updated_at: false,
  expires: false,
  rate_limits: false,
};
