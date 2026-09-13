"use client";

import { ColumnDef } from "@tanstack/react-table";
import { Copy, MoreHorizontal, Pencil, Trash2 } from "lucide-react";

import { CredentialItem } from "@/components/networking";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { DataTableSortHeader } from "@/components/shared/DataTable";
import { IdentityCell } from "@/components/shared/table_cells";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

function CredentialProviderCell({ provider }: { provider: string | undefined }) {
  if (!provider) {
    return <span className="text-sm text-muted-foreground">-</span>;
  }
  const { displayName, logo } = getProviderLogoAndName(provider);
  return (
    <div className="flex items-center gap-2">
      {logo ? (
        <img
          src={logo}
          alt=""
          className="size-4 shrink-0"
          onError={(event) => {
            (event.currentTarget as HTMLImageElement).style.display = "none";
          }}
        />
      ) : null}
      <span className="truncate text-sm">{displayName || provider}</span>
    </div>
  );
}

interface CredentialRowActionsProps {
  credential: CredentialItem;
  onEdit: (credential: CredentialItem) => void;
  onDelete: (credential: CredentialItem) => void;
}

function CredentialRowActions({ credential, onEdit, onDelete }: CredentialRowActionsProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label="Open credential actions"
        data-testid={`credential-actions-${credential.credential_name}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem data-testid="credential-action-edit" onClick={() => onEdit(credential)}>
          <Pencil />
          Edit
        </DropdownMenuItem>
        <DropdownMenuItem
          data-testid="credential-action-copy"
          onClick={() => void copyToClipboard(credential.credential_name, "Credential name copied")}
        >
          <Copy />
          Copy credential name
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          variant="destructive"
          data-testid="credential-action-delete"
          onClick={() => onDelete(credential)}
        >
          <Trash2 />
          Delete
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface CredentialsTableColumnsDeps {
  canModifyCredentials: boolean;
  onEdit: (credential: CredentialItem) => void;
  onDelete: (credential: CredentialItem) => void;
}

export const getCredentialsTableColumns = ({
  canModifyCredentials,
  onEdit,
  onDelete,
}: CredentialsTableColumnsDeps): ColumnDef<CredentialItem>[] => {
  const dataColumns: ColumnDef<CredentialItem>[] = [
    {
      id: "credential_name",
      accessorKey: "credential_name",
      meta: { title: "Credential Name" },
      header: ({ column }) => <DataTableSortHeader column={column} title="Credential Name" />,
      size: 260,
      enableSorting: true,
      cell: ({ row }) => (
        <IdentityCell title={row.original.credential_name} className="max-w-72" titleClassName="font-medium" />
      ),
    },
    {
      id: "provider",
      accessorKey: "credential_info.custom_llm_provider",
      meta: { title: "Provider" },
      header: "Provider",
      size: 200,
      enableSorting: false,
      cell: ({ row }) => <CredentialProviderCell provider={row.original.credential_info?.custom_llm_provider} />,
    },
  ];

  if (!canModifyCredentials) {
    return dataColumns;
  }

  return [
    ...dataColumns,
    {
      id: "actions",
      meta: { className: "text-right", headerClassName: "text-right" },
      header: () => <span className="sr-only">Actions</span>,
      size: 64,
      enableSorting: false,
      enableHiding: false,
      cell: ({ row }) => (
        <div className="flex justify-end">
          <CredentialRowActions credential={row.original} onEdit={onEdit} onDelete={onDelete} />
        </div>
      ),
    },
  ];
};
