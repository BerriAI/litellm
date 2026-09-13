import type { ColumnDef, ColumnFiltersState } from "@tanstack/react-table";
import { Crown, Info, User, UserPlus } from "lucide-react";
import React, { useState } from "react";

import { Member } from "@/components/networking";
import {
  DataTable,
  DataTableFilterDrawer,
  DataTableFilterField,
  DataTableSortHeader,
  DataTableToolbar,
} from "@/components/shared/DataTable";
import { StatusBadge } from "@/components/shared/table_cells";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SimpleTooltip } from "@/components/ui/tooltip";

import TableIconActionButton from "./IconActionButton/TableIconActionButtons/TableIconActionButton";

export type MemberTableSortValue = string | number | null | undefined;

export interface MemberTableColumn {
  title: React.ReactNode;
  key: string;
  render: (member: Member) => React.ReactNode;
  sortValue?: (member: Member) => MemberTableSortValue;
}

export interface MemberTableProps {
  members: Member[];
  canEdit: boolean;
  onEdit: (member: Member) => void;
  onDelete: (member: Member) => void;
  onAddMember?: () => void;
  roleColumnTitle?: string;
  roleTooltip?: string;
  extraColumns?: MemberTableColumn[];
  showDeleteForMember?: (member: Member) => boolean;
  emptyText?: string;
}

const ALL_ROLES = "all";

export const memberRowId = (member: Member): string => member.user_id ?? member.user_email ?? JSON.stringify(member);

export const memberRoleOptions = (members: readonly Member[]): string[] =>
  Array.from(new Set(members.map((member) => member.role).filter((role) => role !== ""))).sort();

const isAdminRole = (role: string): boolean => {
  const normalized = role.toLowerCase();
  return normalized === "admin" || normalized === "org_admin";
};

function RoleHeaderTitle({ title, tooltip }: { title: string; tooltip?: string }) {
  if (tooltip === undefined) return <>{title}</>;
  return (
    <span className="inline-flex items-center gap-2">
      {title}
      <SimpleTooltip content={tooltip}>
        <Info className="size-3.5" />
      </SimpleTooltip>
    </span>
  );
}

const ACTIONS_COLUMN_WIDTH = 120;

interface MemberColumnDeps {
  canEdit: boolean;
  onEdit: (member: Member) => void;
  onDelete: (member: Member) => void;
  roleColumnTitle: string;
  roleTooltip?: string;
  extraColumns: MemberTableColumn[];
  showDeleteForMember?: (member: Member) => boolean;
}

const extraColumnDef = (column: MemberTableColumn): ColumnDef<Member> => {
  const { sortValue } = column;
  if (sortValue === undefined) {
    return {
      id: column.key,
      header: () => <span className="font-medium">{column.title}</span>,
      enableSorting: false,
      enableGlobalFilter: false,
      cell: ({ row }) => column.render(row.original),
    };
  }
  return {
    id: column.key,
    accessorFn: (member) => sortValue(member) ?? undefined,
    header: ({ column: tableColumn }) => <DataTableSortHeader column={tableColumn} title={column.title} />,
    sortDescFirst: false,
    sortUndefined: "last",
    enableGlobalFilter: false,
    cell: ({ row }) => column.render(row.original),
  };
};

const buildColumns = ({
  canEdit,
  onEdit,
  onDelete,
  roleColumnTitle,
  roleTooltip,
  extraColumns,
  showDeleteForMember,
}: MemberColumnDeps): ColumnDef<Member>[] => [
  {
    id: "user_alias",
    accessorFn: (member) => member.user_alias || undefined,
    header: ({ column }) => <DataTableSortHeader column={column} title="Name" />,
    sortingFn: "text",
    sortUndefined: "last",
    enableGlobalFilter: true,
    meta: { title: "Name" },
    cell: ({ row }) => row.original.user_alias || <span className="text-muted-foreground">-</span>,
  },
  {
    id: "user_email",
    accessorFn: (member) => member.user_email || undefined,
    header: ({ column }) => <DataTableSortHeader column={column} title="User Email" />,
    sortingFn: "text",
    sortUndefined: "last",
    enableGlobalFilter: true,
    meta: { title: "User Email" },
    cell: ({ row }) => row.original.user_email || "-",
  },
  {
    id: "user_id",
    accessorFn: (member) => member.user_id ?? undefined,
    header: "User ID",
    enableSorting: false,
    enableGlobalFilter: true,
    cell: ({ row }) =>
      row.original.user_id === "default_user_id" ? (
        <StatusBadge tone="info" label="Default Proxy Admin" />
      ) : (
        row.original.user_id || "-"
      ),
  },
  {
    id: "role",
    accessorFn: (member) => member.role,
    header: ({ column }) => (
      <DataTableSortHeader column={column} title={<RoleHeaderTitle title={roleColumnTitle} tooltip={roleTooltip} />} />
    ),
    sortingFn: "text",
    filterFn: "equalsString",
    enableGlobalFilter: false,
    meta: { title: roleColumnTitle },
    cell: ({ row }) => (
      <span className="inline-flex items-center gap-2">
        {isAdminRole(row.original.role) ? <Crown className="size-3.5" /> : <User className="size-3.5" />}
        <span className="capitalize">{row.original.role || "-"}</span>
      </span>
    ),
  },
  ...extraColumns.map(extraColumnDef),
  {
    id: "actions",
    header: "Actions",
    size: ACTIONS_COLUMN_WIDTH,
    enableSorting: false,
    enableGlobalFilter: false,
    meta: { pinned: "right" },
    cell: ({ row }) =>
      canEdit ? (
        <span className="inline-flex items-center gap-2">
          <TableIconActionButton
            variant="Edit"
            tooltipText="Edit member"
            dataTestId="edit-member"
            onClick={() => onEdit(row.original)}
          />
          {(!showDeleteForMember || showDeleteForMember(row.original)) && (
            <TableIconActionButton
              variant="Delete"
              tooltipText="Delete member"
              dataTestId="delete-member"
              onClick={() => onDelete(row.original)}
            />
          )}
        </span>
      ) : null,
  },
];

export default function MemberTable({
  members,
  canEdit,
  onEdit,
  onDelete,
  onAddMember,
  roleColumnTitle = "Role",
  roleTooltip,
  extraColumns = [],
  showDeleteForMember,
  emptyText,
}: MemberTableProps) {
  const [globalFilter, setGlobalFilter] = useState("");
  const [columnFilters, setColumnFilters] = useState<ColumnFiltersState>([]);
  const [filtersOpen, setFiltersOpen] = useState(false);

  const columnDeps: MemberColumnDeps = {
    canEdit,
    onEdit,
    onDelete,
    roleColumnTitle,
    roleTooltip,
    extraColumns,
    showDeleteForMember,
  };
  const columns = buildColumns(columnDeps);
  const roleFilterItems = [
    { value: ALL_ROLES, label: "All Roles" },
    ...memberRoleOptions(members).map((role) => ({ value: role, label: role })),
  ];

  const isNarrowed = globalFilter !== "" || columnFilters.length > 0;

  return (
    <div className="flex w-full flex-col gap-2">
      <span className="inline-flex text-sm text-foreground">
        {members.length} Member{members.length !== 1 ? "s" : ""}
      </span>
      <DataTable
        data={members}
        columns={columns}
        getRowId={memberRowId}
        sortingMode="client"
        defaultSorting={[{ id: "user_alias", desc: false }]}
        filterMode="client"
        columnFilters={columnFilters}
        onColumnFiltersChange={setColumnFilters}
        globalFilter={globalFilter}
        onGlobalFilterChange={setGlobalFilter}
        noDataMessage={
          <span className="text-muted-foreground">
            {isNarrowed ? "No members match your search or filters" : emptyText ?? "No data"}
          </span>
        }
        toolbar={(table) => (
          <>
            <DataTableToolbar
              table={table}
              searchValue={globalFilter}
              onSearchChange={setGlobalFilter}
              searchPlaceholder="Search by name, email, or user ID"
              onOpenFilters={() => setFiltersOpen(true)}
              showViewOptions={false}
            />
            <DataTableFilterDrawer
              table={table}
              open={filtersOpen}
              onOpenChange={setFiltersOpen}
              title="Filters"
              description="Narrow down members"
            >
              {({ get, set }) => (
                <DataTableFilterField label={roleColumnTitle}>
                  <Select
                    items={roleFilterItems}
                    value={(get("role") as string | undefined) ?? ALL_ROLES}
                    onValueChange={(value) => set("role", value === ALL_ROLES ? undefined : value)}
                  >
                    <SelectTrigger className="w-full" data-testid="filter-role">
                      <SelectValue placeholder="All Roles" />
                    </SelectTrigger>
                    <SelectContent>
                      {roleFilterItems.map((item) => (
                        <SelectItem key={item.value} value={item.value}>
                          {item.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </DataTableFilterField>
              )}
            </DataTableFilterDrawer>
          </>
        )}
      />
      {onAddMember && canEdit && (
        <Button onClick={onAddMember} className="self-start">
          <UserPlus className="size-4" />
          Add Member
        </Button>
      )}
    </div>
  );
}
