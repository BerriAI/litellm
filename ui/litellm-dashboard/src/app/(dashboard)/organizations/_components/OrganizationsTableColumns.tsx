"use client";

import { ColumnDef } from "@tanstack/react-table";
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react";

import { DataTableMultiSortHeader, DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdentityCell, ModelsCell, MoneyCell } from "@/components/shared/table_cells";
import { Organization } from "@/components/networking";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";

interface OrganizationBudget {
  max_budget?: number | null;
  tpm_limit?: number | null;
  rpm_limit?: number | null;
}

const getOrganizationBudget = (organization: Organization): OrganizationBudget =>
  (organization.litellm_budget_table ?? {}) as OrganizationBudget;

function OrganizationLimitsCell({ organization }: { organization: Organization }) {
  const { tpm_limit, rpm_limit } = getOrganizationBudget(organization);
  return (
    <div className="flex flex-col text-xs text-muted-foreground">
      <span>TPM: {tpm_limit ?? "Unlimited"}</span>
      <span>RPM: {rpm_limit ?? "Unlimited"}</span>
    </div>
  );
}

interface OrganizationRowActionsProps {
  organization: Organization;
  onEditClick: (organizationId: string) => void;
  onDeleteClick: (organizationId: string) => void;
}

function OrganizationRowActions({ organization, onEditClick, onDeleteClick }: OrganizationRowActionsProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label="Open organization actions"
        data-testid={`organization-actions-${organization.organization_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          data-testid="organization-action-edit"
          onClick={() => onEditClick(organization.organization_id)}
        >
          <Pencil />
          Edit
        </DropdownMenuItem>
        <DropdownMenuItem
          variant="destructive"
          data-testid="organization-action-delete"
          onClick={() => onDeleteClick(organization.organization_id)}
        >
          <Trash2 />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

export interface OrganizationsTableColumnsDeps {
  userRole: string;
  onOrganizationClick: (organizationId: string) => void;
  onEditClick: (organizationId: string) => void;
  onDeleteClick: (organizationId: string) => void;
}

export const getOrganizationsTableColumns = ({
  userRole,
  onOrganizationClick,
  onEditClick,
  onDeleteClick,
}: OrganizationsTableColumnsDeps): ColumnDef<Organization>[] => [
  {
    id: "organization_id",
    accessorKey: "organization_id",
    meta: { title: "Organization ID" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Organization ID" />,
    size: 220,
    enableSorting: true,
    cell: ({ row }) => (
      <IdentityCell
        title={row.original.organization_id}
        titleClassName="font-mono text-xs font-normal"
        className="max-w-56"
        onClick={() => onOrganizationClick(row.original.organization_id)}
      />
    ),
  },
  {
    id: "organization_alias",
    accessorKey: "organization_alias",
    meta: { title: "Organization Name" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Organization Name" />,
    size: 200,
    enableSorting: true,
    cell: ({ row }) => {
      const alias = row.original.organization_alias;
      return (
        <span className="block max-w-56 truncate text-sm font-medium" title={alias ?? undefined}>
          {alias || "-"}
        </span>
      );
    },
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    sortingFn: "datetime",
    meta: { title: "Created" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Created" />,
    size: 130,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} precision="date" />,
  },
  {
    id: "spend",
    accessorKey: "spend",
    meta: { title: "Spend (USD)", numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title="Spend (USD)" />,
    size: 120,
    enableSorting: true,
    cell: ({ row }) => <MoneyCell value={row.original.spend} decimals={4} />,
  },
  {
    id: "max_budget",
    accessorFn: (organization) => getOrganizationBudget(organization).max_budget ?? Number.POSITIVE_INFINITY,
    sortUndefined: "last",
    sortDescFirst: true,
    meta: { title: "Budget (USD)", numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title="Budget (USD)" />,
    size: 120,
    enableSorting: true,
    cell: ({ row }) => (
      <MoneyCell value={getOrganizationBudget(row.original).max_budget} decimals={2} emptyText="Unlimited" showZero />
    ),
  },
  {
    id: "models",
    accessorFn: (organization) =>
      organization.models?.length && !organization.models.includes("all-proxy-models")
        ? [...organization.models].sort().join(", ")
        : undefined,
    sortUndefined: "last",
    sortDescFirst: false,
    meta: { title: "Models", skeleton: "chips" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Models" />,
    size: 260,
    enableSorting: true,
    cell: ({ row }) => <ModelsCell models={row.original.models} />,
  },
  {
    id: "limits",
    meta: { title: "TPM / RPM Limits" },
    header: ({ table }) => (
      <DataTableMultiSortHeader
        table={table}
        fields={[
          { id: "tpm_limit", label: "TPM" },
          { id: "rpm_limit", label: "RPM" },
        ]}
      />
    ),
    size: 150,
    enableSorting: false,
    cell: ({ row }) => <OrganizationLimitsCell organization={row.original} />,
  },
  {
    id: "members",
    accessorFn: (organization) => organization.members?.length ?? 0,
    sortUndefined: "last",
    sortDescFirst: true,
    meta: { title: "Members", numeric: true },
    header: ({ column }) => <DataTableSortHeader column={column} title="Members" />,
    size: 100,
    enableSorting: true,
    cell: ({ row }) => <span className="text-sm">{row.original.members?.length ?? 0} Members</span>,
  },
  ...(["tpm_limit", "rpm_limit"] as const).map(
    (field): ColumnDef<Organization> => ({
      id: field,
      accessorFn: (organization) => getOrganizationBudget(organization)[field] ?? Number.POSITIVE_INFINITY,
      sortUndefined: "last",
      sortDescFirst: true,
      enableHiding: false,
    }),
  ),
  {
    id: "actions",
    meta: { className: "text-right", headerClassName: "text-right" },
    header: () => <span className="sr-only">Actions</span>,
    size: 64,
    enableSorting: false,
    enableHiding: false,
    cell: ({ row }) =>
      userRole === "Admin" ? (
        <div className="flex justify-end">
          <OrganizationRowActions organization={row.original} onEditClick={onEditClick} onDeleteClick={onDeleteClick} />
        </div>
      ) : null,
  },
];
