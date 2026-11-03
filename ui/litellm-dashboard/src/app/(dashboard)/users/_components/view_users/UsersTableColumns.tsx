"use client";

import { ColumnDef } from "@tanstack/react-table";
import type { TFunction } from "i18next";
import { ChartColumn, Copy, Info, KeyRound, MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import Link from "next/link";
import { useTranslation } from "react-i18next";

import { UserInfo } from "@/components/networking";
import { createSelectionColumn, DataTableSortHeader } from "@/components/shared/DataTable";
import { CellTooltip, DateCell, IdentityCell, MoneyCell, StatusBadge } from "@/components/shared/table_cells";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";
import { usageHrefForUser } from "@/app/(dashboard)/usage/_components/usageUrlState";
import { copyToClipboard } from "@/utils/dataUtils";

function isScimInactive(user: UserInfo): boolean {
  return (user.metadata as Record<string, unknown> | null | undefined)?.scim_active === false;
}

interface UserRowActionsProps {
  user: UserInfo;
  canViewUserUsage: boolean;
  onUserClick: (userId: string, openInEditMode?: boolean) => void;
  onDeleteUser: (user: UserInfo) => void;
  onResetPassword: (userId: string) => void;
}

function UserRowActions({ user, canViewUserUsage, onUserClick, onDeleteUser, onResetPassword }: UserRowActionsProps) {
  const { t } = useTranslation();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("users.table.rowActions.open")}
        data-testid={`user-actions-${user.user_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-48">
        <DropdownMenuItem onClick={() => onUserClick(user.user_id, true)} data-testid="user-action-edit">
          <Pencil />
          {t("users.table.rowActions.edit")}
        </DropdownMenuItem>
        <DropdownMenuItem onClick={() => onResetPassword(user.user_id)} data-testid="user-action-reset-password">
          <KeyRound />
          {t("users.table.rowActions.resetPassword")}
        </DropdownMenuItem>
        <DropdownMenuItem
          onClick={() => void copyToClipboard(user.user_id, t("users.table.rowActions.copied"))}
          data-testid="user-action-copy"
        >
          <Copy />
          {t("users.table.rowActions.copyId")}
        </DropdownMenuItem>
        {canViewUserUsage && (
          <DropdownMenuItem
            render={<Link href={usageHrefForUser(user.user_id)} />}
            data-testid="user-action-view-usage"
          >
            <ChartColumn />
            {t("users.table.rowActions.viewUsage")}
          </DropdownMenuItem>
        )}
        <DropdownMenuSeparator />
        <DropdownMenuItem variant="destructive" onClick={() => onDeleteUser(user)} data-testid="user-action-delete">
          <Trash2 />
          {t("users.table.rowActions.delete")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export interface UsersTableColumnsDeps {
  possibleUIRoles: Record<string, Record<string, string>> | null;
  includeSelection: boolean;
  canViewUserUsage: boolean;
  onUserClick: (userId: string, openInEditMode?: boolean) => void;
  onDeleteUser: (user: UserInfo) => void;
  onResetPassword: (userId: string) => void;
  t: TFunction;
}

export const getUsersTableColumns = ({
  possibleUIRoles,
  includeSelection,
  canViewUserUsage,
  onUserClick,
  onDeleteUser,
  onResetPassword,
  t,
}: UsersTableColumnsDeps): ColumnDef<UserInfo>[] => {
  const baseColumns: ColumnDef<UserInfo>[] = [
    {
      id: "user_id",
      accessorKey: "user_id",
      meta: { title: t("users.table.columns.userId") },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={t("users.table.columns.userId")} variant="header-cycle" />
      ),
      size: 220,
      enableSorting: true,
      cell: ({ row }) => (
        <IdentityCell
          title={row.original.user_id}
          titleClassName="font-mono text-xs text-primary"
          onClick={() => onUserClick(row.original.user_id, false)}
        />
      ),
    },
    {
      id: "user_email",
      accessorKey: "user_email",
      meta: { title: t("users.table.columns.email") },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={t("users.table.columns.email")} variant="header-cycle" />
      ),
      size: 220,
      enableSorting: true,
      cell: ({ row }) => (
        <span className="block max-w-60 truncate text-sm" title={row.original.user_email ?? undefined}>
          {row.original.user_email || "-"}
        </span>
      ),
    },
    {
      id: "status",
      accessorFn: (user) => (isScimInactive(user) ? "Inactive" : "Active"),
      meta: { title: t("users.table.columns.status"), skeleton: "badge" },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("users.table.columns.status")} />,
      sortDescFirst: false,
      size: 110,
      enableSorting: true,
      cell: ({ row }) => {
        if (isScimInactive(row.original)) {
          return (
            <StatusBadge
              tone="error"
              label={t("users.table.status.inactive")}
              tooltip={t("users.table.status.scimInactiveHint")}
              dataTestId={`user-status-${row.original.user_id}`}
            />
          );
        }
        return (
          <StatusBadge
            tone="success"
            label={t("users.table.status.active")}
            dataTestId={`user-status-${row.original.user_id}`}
          />
        );
      },
    },
    {
      id: "user_role",
      accessorKey: "user_role",
      meta: { title: t("users.table.columns.globalProxyRole") },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={t("users.table.columns.globalProxyRole")} variant="header-cycle" />
      ),
      size: 160,
      enableSorting: true,
      cell: ({ row }) => (
        <span className="text-sm">
          {(row.original.user_role && possibleUIRoles?.[row.original.user_role]?.ui_label) || "-"}
        </span>
      ),
    },
    {
      id: "user_alias",
      accessorKey: "user_alias",
      meta: { title: t("users.table.columns.userAlias") },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("users.table.columns.userAlias")} />,
      sortDescFirst: false,
      size: 150,
      enableSorting: true,
      cell: ({ row }) => (
        <span className="block max-w-40 truncate text-sm" title={row.original.user_alias ?? undefined}>
          {row.original.user_alias || "-"}
        </span>
      ),
    },
    {
      id: "spend",
      accessorKey: "spend",
      meta: { title: t("users.table.columns.spend"), numeric: true },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={t("users.table.columns.spend")} variant="header-cycle" />
      ),
      size: 130,
      enableSorting: true,
      cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={2} />,
    },
    {
      id: "max_budget",
      accessorKey: "max_budget",
      meta: { title: t("users.table.columns.budget"), numeric: true },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("users.table.columns.budget")} />,
      sortDescFirst: true,
      size: 130,
      enableSorting: true,
      cell: ({ row }) => (
        <MoneyCell value={row.original.max_budget} decimals={2} emptyText={t("users.table.unlimitedBudget")} showZero />
      ),
    },
    {
      id: "sso_user_id",
      accessorKey: "sso_user_id",
      meta: { title: t("users.table.columns.ssoId") },
      header: ({ column }) => (
        <span className="flex items-center gap-1.5">
          <DataTableSortHeader column={column} title={t("users.table.columns.ssoId")} />
          <CellTooltip
            content={t("users.table.ssoIdHint")}
            trigger={
              <Info className="size-3.5 shrink-0 text-muted-foreground" aria-label={t("users.table.ssoIdHintLabel")} />
            }
          />
        </span>
      ),
      sortDescFirst: false,
      size: 160,
      enableSorting: true,
      cell: ({ row }) => (
        <span className="block max-w-40 truncate font-mono text-xs" title={row.original.sso_user_id ?? undefined}>
          {row.original.sso_user_id ?? "-"}
        </span>
      ),
    },
    {
      id: "key_count",
      accessorKey: "key_count",
      meta: { title: t("users.table.columns.virtualKeys"), skeleton: "badge" },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("users.table.columns.virtualKeys")} />,
      sortDescFirst: true,
      size: 120,
      enableSorting: true,
      cell: ({ row }) => {
        const keyCount = row.original.key_count;
        if (keyCount > 0) {
          return (
            <Badge
              variant="outline"
              className="whitespace-nowrap border-indigo-200 bg-indigo-50 font-normal text-indigo-600 dark:border-indigo-800 dark:bg-indigo-950 dark:text-indigo-300"
            >
              {keyCount} {t("users.table.keyCount", { count: keyCount })}
            </Badge>
          );
        }
        return (
          <Badge
            variant="outline"
            className="whitespace-nowrap border-border bg-muted font-normal text-muted-foreground"
          >
            {t("users.table.noKeys")}
          </Badge>
        );
      },
    },
    {
      id: "created_at",
      accessorKey: "created_at",
      meta: { title: t("users.table.columns.createdAt") },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={t("users.table.columns.createdAt")} variant="header-cycle" />
      ),
      size: 130,
      enableSorting: true,
      cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
    },
    {
      id: "updated_at",
      accessorKey: "updated_at",
      meta: { title: t("users.table.columns.updatedAt") },
      header: ({ column }) => <DataTableSortHeader column={column} title={t("users.table.columns.updatedAt")} />,
      sortDescFirst: false,
      size: 130,
      enableSorting: true,
      cell: ({ row }) => <DateCell value={row.original.updated_at} precision="date" />,
    },
    {
      id: "actions",
      meta: { title: t("users.table.columns.actions"), className: "text-right", headerClassName: "text-right" },
      header: () => <span className="sr-only">{t("users.table.columns.actions")}</span>,
      size: 60,
      enableSorting: false,
      enableHiding: false,
      cell: ({ row }) => (
        <div className="flex justify-end">
          <UserRowActions
            user={row.original}
            canViewUserUsage={canViewUserUsage}
            onUserClick={onUserClick}
            onDeleteUser={onDeleteUser}
            onResetPassword={onResetPassword}
          />
        </div>
      ),
    },
  ];

  if (!includeSelection) {
    return baseColumns;
  }

  return [
    createSelectionColumn<UserInfo>({
      rowAriaLabel: (row) => t("users.table.selectRow", { user: row.original.user_email || row.original.user_id }),
    }),
    ...baseColumns,
  ];
};
