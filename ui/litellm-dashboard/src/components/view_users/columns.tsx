import { ColumnDef } from "@tanstack/react-table";
import { Badge, Grid, Icon } from "@tremor/react";
import { Tooltip, Checkbox } from "antd";
import { UserInfo } from "./types";
import { PencilAltIcon, TrashIcon, InformationCircleIcon, RefreshIcon } from "@heroicons/react/outline";
import { formatNumberWithCommas } from "@/utils/dataUtils";

interface SelectionOptions {
  selectedUsers: UserInfo[];
  onSelectUser: (user: UserInfo, isSelected: boolean) => void;
  onSelectAll: (isSelected: boolean) => void;
  isUserSelected: (user: UserInfo) => boolean;
  isAllSelected: boolean;
  isIndeterminate: boolean;
}

export const columns = (
  possibleUIRoles: Record<string, Record<string, string>>,
  handleEdit: (user: UserInfo) => void,
  handleDelete: (userId: string) => void,
  handleResetPassword: (userId: string) => void,
  handleUserClick: (userId: string, openInEditMode?: boolean) => void,
  selectionOptions?: SelectionOptions,
): ColumnDef<UserInfo>[] => {
  const baseColumns: ColumnDef<UserInfo>[] = [
    {
      header: "User ID",
      accessorKey: "user_id",
      cell: ({ row }) => (
        <Tooltip title={row.original.user_id}>
          <span className="text-xs">{row.original.user_id ? `${row.original.user_id.slice(0, 7)}...` : "-"}</span>
        </Tooltip>
      ),
    },
    {
      header: "Email",
      accessorKey: "user_email",
      cell: ({ row }) => <span className="text-xs">{row.original.user_email || "-"}</span>,
    },
    {
      header: "Global Proxy Role",
      accessorKey: "user_role",
      cell: ({ row }) => <span className="text-xs">{possibleUIRoles?.[row.original.user_role]?.ui_label || "-"}</span>,
    },
    {
      header: "Spend (USD)",
      accessorKey: "spend",
      cell: ({ row }) => (
        <span className="text-xs">{row.original.spend ? formatNumberWithCommas(row.original.spend, 4) : "-"}</span>
      ),
    },
    {
      header: "Budget (USD)",
      accessorKey: "max_budget",
      cell: ({ row }) => (
        <span className="text-xs">{row.original.max_budget !== null ? row.original.max_budget : "Unlimited"}</span>
      ),
    },
    {
      header: () => (
        <div className="flex items-center gap-2">
          <span>SSO ID</span>
          <Tooltip title="SSO ID is the ID of the user in the SSO provider. If the user is not using SSO, this will be null.">
            <InformationCircleIcon className="w-4 h-4" />
          </Tooltip>
        </div>
      ),
      accessorKey: "sso_user_id",
      cell: ({ row }) => (
        <span className="text-xs">{row.original.sso_user_id !== null ? row.original.sso_user_id : "-"}</span>
      ),
    },
    {
      header: "API Keys",
      accessorKey: "key_count",
      cell: ({ row }) => (
        <Grid numItems={2}>
          {row.original.key_count > 0 ? (
            <Badge size="xs" color="indigo">
              {row.original.key_count} Keys
            </Badge>
          ) : (
            <Badge size="xs" color="gray">
              No Keys
            </Badge>
          )}
        </Grid>
      ),
    },
    {
      header: "Created At",
      accessorKey: "created_at",
      sortingFn: "datetime",
      cell: ({ row }) => (
        <span className="text-xs">
          {row.original.created_at ? new Date(row.original.created_at).toLocaleDateString() : "-"}
        </span>
      ),
    },
    {
      header: "Updated At",
      accessorKey: "updated_at",
      sortingFn: "datetime",
      cell: ({ row }) => (
        <span className="text-xs">
          {row.original.updated_at ? new Date(row.original.updated_at).toLocaleDateString() : "-"}
        </span>
      ),
    },
    {
      id: "actions",
      header: "",
      cell: ({ row }) => (
        <div className="flex gap-2">
          <Tooltip title="Edit user details" zIndex={9999}>
            <Icon icon={PencilAltIcon} size="sm" onClick={() => handleUserClick(row.original.user_id, true)} />
          </Tooltip>
          <Tooltip title="Delete user" zIndex={9999}>
            <Icon icon={TrashIcon} size="sm" onClick={() => handleDelete(row.original.user_id)} />
          </Tooltip>
          <Tooltip title="Reset Password" zIndex={9999}>
            <Icon icon={RefreshIcon} size="sm" onClick={() => handleResetPassword(row.original.user_id)} />
          </Tooltip>
        </div>
      ),
    },
  ];

  // Add selection column if selection is enabled
  if (selectionOptions) {
    const { onSelectUser, onSelectAll, isUserSelected, isAllSelected, isIndeterminate } = selectionOptions;

    return [
      {
        id: "select",
        header: () => (
          <Checkbox
            indeterminate={isIndeterminate}
            checked={isAllSelected}
            onChange={(e) => onSelectAll(e.target.checked)}
            onClick={(e) => e.stopPropagation()}
          />
        ),
        cell: ({ row }) => (
          <Checkbox
            checked={isUserSelected(row.original)}
            onChange={(e) => onSelectUser(row.original, e.target.checked)}
            onClick={(e) => e.stopPropagation()}
          />
        ),
      },
      ...baseColumns,
    ];
  }

  return baseColumns;
};
