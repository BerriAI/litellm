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
  handleDelete: (user: UserInfo) => void,
  handleResetPassword: (userId: string) => void,
  handleUserClick: (userId: string, openInEditMode?: boolean) => void,
  selectionOptions?: SelectionOptions,
): ColumnDef<UserInfo>[] => {
  // Backend sortable columns: user_id, user_email, created_at, spend, user_alias, user_role
  const baseColumns: ColumnDef<UserInfo>[] = [
    {
      header: "User ID",
      accessorKey: "user_id",
      enableSorting: true,
      cell: ({ row }) => (
        <Tooltip title={row.original.user_id}>
          <span className="text-xs">{row.original.user_id ? `${row.original.user_id.slice(0, 7)}...` : "-"}</span>
        </Tooltip>
      ),
    },
    {
      header: "Email",
      accessorKey: "user_email",
      enableSorting: true,
      cell: ({ row }) => <span className="text-xs">{row.original.user_email || "-"}</span>,
    },
    {
      header: "Global Proxy Role",
      accessorKey: "user_role",
      enableSorting: true,
      cell: ({ row }) => <span className="text-xs">{possibleUIRoles?.[row.original.user_role]?.ui_label || "-"}</span>,
    },
    {
      header: "User Alias",
      accessorKey: "user_alias",
      enableSorting: false,
      cell: ({ row }) => <span className="text-xs">{row.original.user_alias || "-"}</span>,
    },
    {
      header: "Spend (USD)",
      accessorKey: "spend",
      enableSorting: true,
      cell: ({ row }) => (
        <span className="text-xs">{row.original.spend ? formatNumberWithCommas(row.original.spend, 4) : "-"}</span>
      ),
    },
    {
      header: "Budget (USD)",
      accessorKey: "max_budget",
      enableSorting: false,
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
      enableSorting: false,
      cell: ({ row }) => (
        <span className="text-xs">{row.original.sso_user_id !== null ? row.original.sso_user_id : "-"}</span>
      ),
    },
    {
      header: "Virtual Keys",
      accessorKey: "key_count",
      enableSorting: false,
      cell: ({ row }) => (
        <Grid numItems={2}>
          {row.original.key_count > 0 ? (
            <Badge size="xs" color="indigo">
              {row.original.key_count} {row.original.key_count === 1 ? "Key" : "Keys"}
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
      enableSorting: true,
      cell: ({ row }) => (
        <span className="text-xs">
          {row.original.created_at ? new Date(row.original.created_at).toLocaleDateString() : "-"}
        </span>
      ),
    },
    {
      header: "Updated At",
      accessorKey: "updated_at",
      enableSorting: false,
      cell: ({ row }) => (
        <span className="text-xs">
          {row.original.updated_at ? new Date(row.original.updated_at).toLocaleDateString() : "-"}
        </span>
      ),
    },
    {
      id: "actions",
      header: "Actions",
      enableSorting: false,
      cell: ({ row }) => (
        <div className="flex gap-2">
          <Tooltip title="Edit user details">
            <Icon
              icon={PencilAltIcon}
              size="sm"
              onClick={() => handleUserClick(row.original.user_id, true)}
              className="cursor-pointer hover:text-blue-600"
            />
          </Tooltip>
          <Tooltip title="Delete user">
            <Icon
              icon={TrashIcon}
              size="sm"
              onClick={() => handleDelete(row.original)}
              className="cursor-pointer hover:text-red-600"
            />
          </Tooltip>
          <Tooltip title="Reset Password">
            <Icon
              icon={RefreshIcon}
              size="sm"
              onClick={() => handleResetPassword(row.original.user_id)}
              className="cursor-pointer hover:text-green-600"
            />
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
        enableSorting: false,
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
