import { ColumnDef } from "@tanstack/react-table";
import { Badge } from "@/components/ui/badge";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Copy, Info, Pencil, RefreshCcw, Trash2 } from "lucide-react";
import { UserInfo } from "./types";
import { formatNumberWithCommas, copyToClipboard } from "@/utils/dataUtils";

interface SelectionOptions {
  selectedUsers: UserInfo[];
  onSelectUser: (user: UserInfo, isSelected: boolean) => void;
  onSelectAll: (isSelected: boolean) => void;
  isUserSelected: (user: UserInfo) => boolean;
  isAllSelected: boolean;
  isIndeterminate: boolean;
}

const IconTipButton: React.FC<{
  tooltip: string;
  onClick: () => void;
  icon: React.ReactNode;
  className?: string;
  ariaLabel?: string;
}> = ({ tooltip, onClick, icon, className, ariaLabel }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onClick}
          className={cn(
            "cursor-pointer text-muted-foreground hover:text-foreground",
            className,
          )}
          aria-label={ariaLabel ?? tooltip}
        >
          {icon}
        </button>
      </TooltipTrigger>
      <TooltipContent>{tooltip}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

export const columns = (
  possibleUIRoles: Record<string, Record<string, string>>,
  handleEdit: (user: UserInfo) => void,
  handleDelete: (user: UserInfo) => void,
  handleResetPassword: (userId: string) => void,
  handleUserClick: (userId: string, openInEditMode?: boolean) => void,
  selectionOptions?: SelectionOptions,
): ColumnDef<UserInfo>[] => {
  const baseColumns: ColumnDef<UserInfo>[] = [
    {
      header: "User ID",
      accessorKey: "user_id",
      enableSorting: true,
      cell: ({ row }) => (
        <div className="flex items-center space-x-2">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <span className="text-xs">
                  {row.original.user_id
                    ? `${row.original.user_id.slice(0, 7)}...`
                    : "-"}
                </span>
              </TooltipTrigger>
              <TooltipContent>{row.original.user_id}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
          {row.original.user_id && (
            <IconTipButton
              tooltip="Copy User ID"
              onClick={() =>
                copyToClipboard(
                  row.original.user_id,
                  "User ID copied to clipboard",
                )
              }
              icon={<Copy className="h-3 w-3" />}
              className="hover:text-primary"
            />
          )}
        </div>
      ),
    },
    {
      header: "Email",
      accessorKey: "user_email",
      enableSorting: true,
      cell: ({ row }) => (
        <span className="text-xs">{row.original.user_email || "-"}</span>
      ),
    },
    {
      header: "Global Proxy Role",
      accessorKey: "user_role",
      enableSorting: true,
      cell: ({ row }) => (
        <span className="text-xs">
          {possibleUIRoles?.[row.original.user_role]?.ui_label || "-"}
        </span>
      ),
    },
    {
      header: "User Alias",
      accessorKey: "user_alias",
      enableSorting: false,
      cell: ({ row }) => (
        <span className="text-xs">{row.original.user_alias || "-"}</span>
      ),
    },
    {
      header: "Spend (USD)",
      accessorKey: "spend",
      enableSorting: true,
      cell: ({ row }) => (
        <span className="text-xs">
          {row.original.spend
            ? formatNumberWithCommas(row.original.spend, 4)
            : "-"}
        </span>
      ),
    },
    {
      header: "Budget (USD)",
      accessorKey: "max_budget",
      enableSorting: false,
      cell: ({ row }) => (
        <span className="text-xs">
          {row.original.max_budget !== null
            ? row.original.max_budget
            : "Unlimited"}
        </span>
      ),
    },
    {
      header: () => (
        <div className="flex items-center gap-2">
          <span>SSO ID</span>
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <Info className="h-3.5 w-3.5" />
              </TooltipTrigger>
              <TooltipContent className="max-w-xs">
                SSO ID is the ID of the user in the SSO provider. If the user
                is not using SSO, this will be null.
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      ),
      accessorKey: "sso_user_id",
      enableSorting: false,
      cell: ({ row }) => (
        <span className="text-xs">
          {row.original.sso_user_id !== null ? row.original.sso_user_id : "-"}
        </span>
      ),
    },
    {
      header: "Virtual Keys",
      accessorKey: "key_count",
      enableSorting: false,
      cell: ({ row }) => (
        <div className="grid grid-cols-2">
          {row.original.key_count > 0 ? (
            <Badge className="text-xs bg-indigo-100 text-indigo-700 dark:bg-indigo-950 dark:text-indigo-300">
              {row.original.key_count}{" "}
              {row.original.key_count === 1 ? "Key" : "Keys"}
            </Badge>
          ) : (
            <Badge className="text-xs bg-muted text-muted-foreground">
              No Keys
            </Badge>
          )}
        </div>
      ),
    },
    {
      header: "Created At",
      accessorKey: "created_at",
      enableSorting: true,
      cell: ({ row }) => (
        <span className="text-xs">
          {row.original.created_at
            ? new Date(row.original.created_at).toLocaleDateString()
            : "-"}
        </span>
      ),
    },
    {
      header: "Updated At",
      accessorKey: "updated_at",
      enableSorting: false,
      cell: ({ row }) => (
        <span className="text-xs">
          {row.original.updated_at
            ? new Date(row.original.updated_at).toLocaleDateString()
            : "-"}
        </span>
      ),
    },
    {
      id: "actions",
      header: "Actions",
      enableSorting: false,
      cell: ({ row }) => (
        <div className="flex gap-2">
          <IconTipButton
            tooltip="Edit user details"
            onClick={() => handleUserClick(row.original.user_id, true)}
            icon={<Pencil className="h-4 w-4" />}
            className="hover:text-primary"
          />
          <IconTipButton
            tooltip="Delete user"
            onClick={() => handleDelete(row.original)}
            icon={<Trash2 className="h-4 w-4" />}
            className="hover:text-destructive"
          />
          <IconTipButton
            tooltip="Reset Password"
            onClick={() => handleResetPassword(row.original.user_id)}
            icon={<RefreshCcw className="h-4 w-4" />}
            className="hover:text-emerald-600 dark:hover:text-emerald-400"
          />
        </div>
      ),
    },
  ];

  if (selectionOptions) {
    const {
      onSelectUser,
      onSelectAll,
      isUserSelected,
      isAllSelected,
      isIndeterminate,
    } = selectionOptions;

    return [
      {
        id: "select",
        enableSorting: false,
        header: () => (
          <Checkbox
            checked={
              isIndeterminate ? "indeterminate" : isAllSelected ? true : false
            }
            onCheckedChange={(c) => onSelectAll(c === true)}
            onClick={(e) => e.stopPropagation()}
          />
        ),
        cell: ({ row }) => (
          <Checkbox
            checked={isUserSelected(row.original)}
            onCheckedChange={(c) => onSelectUser(row.original, c === true)}
            onClick={(e) => e.stopPropagation()}
          />
        ),
      },
      ...baseColumns,
    ];
  }

  return baseColumns;
};
