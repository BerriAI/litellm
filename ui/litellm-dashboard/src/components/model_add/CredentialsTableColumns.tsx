"use client";

import { ColumnDef } from "@tanstack/react-table";
import type { TFunction } from "i18next";
import { useTranslation } from "react-i18next";
import { Copy, MoreHorizontal, Pencil, Trash2 } from "lucide-react";

import { CredentialItem } from "@/components/networking";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
import { DataTableSortHeader } from "@/components/shared/DataTable";
import { credentialLabel } from "@/components/shared/credentialOptions";
import { IdentityCell, StatusBadge } from "@/components/shared/table_cells";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import i18n from "@/i18n/i18n";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";

import { inferAuthMethod } from "./credential_federation";

function CredentialProviderCell({ provider, federated }: { provider: string | undefined; federated: boolean }) {
  const { t } = useTranslation();
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
      {federated && <Badge variant="secondary">{t("modelAdd.credentialsTableColumns.federated")}</Badge>}
    </div>
  );
}

interface CredentialRowActionsProps {
  credential: CredentialItem;
  onEdit: (credential: CredentialItem) => void;
  onDelete: (credential: CredentialItem) => void;
}

function CredentialRowActions({ credential, onEdit, onDelete }: CredentialRowActionsProps) {
  const { t } = useTranslation();
  const configOwned = credential.source === "config";
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("modelAdd.credentialsTableColumns.openActions")}
        data-testid={`credential-actions-${credential.credential_name}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        {configOwned && (
          <>
            <DropdownMenuGroup>
              <DropdownMenuLabel data-testid="credential-config-owned-hint">
                {t("modelAdd.credentialsTableColumns.configOwnedHint")}
              </DropdownMenuLabel>
            </DropdownMenuGroup>
            <DropdownMenuSeparator />
          </>
        )}
        <DropdownMenuItem
          data-testid="credential-action-edit"
          disabled={configOwned}
          onClick={() => onEdit(credential)}
        >
          <Pencil />
          {t("modelAdd.credentialsTableColumns.edit")}
        </DropdownMenuItem>
        <DropdownMenuItem
          data-testid="credential-action-copy"
          onClick={() =>
            void copyToClipboard(credential.credential_name, t("modelAdd.credentialsTableColumns.nameCopied"))
          }
        >
          <Copy />
          {t("modelAdd.credentialsTableColumns.copyName")}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          variant="destructive"
          data-testid="credential-action-delete"
          disabled={configOwned}
          onClick={() => onDelete(credential)}
        >
          <Trash2 />
          {t("modelAdd.credentialsTableColumns.delete")}
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

export const getCredentialsTableColumns = (
  { canModifyCredentials, onEdit, onDelete }: CredentialsTableColumnsDeps,
  t: TFunction = i18n.t.bind(i18n),
): ColumnDef<CredentialItem>[] => {
  const dataColumns: ColumnDef<CredentialItem>[] = [
    {
      id: "credential_name",
      accessorFn: credentialLabel,
      meta: { title: t("modelAdd.credentialsTableColumns.credentialName") },
      header: ({ column }) => (
        <DataTableSortHeader column={column} title={t("modelAdd.credentialsTableColumns.credentialName")} />
      ),
      size: 260,
      enableSorting: true,
      cell: ({ row }) => (
        <IdentityCell
          title={credentialLabel(row.original)}
          subtitle={row.original.display_name ? row.original.credential_name : undefined}
          className="max-w-72"
          titleClassName="font-medium"
        />
      ),
    },
    {
      id: "provider",
      accessorKey: "credential_info.custom_llm_provider",
      meta: { title: t("modelAdd.credentialsTableColumns.provider") },
      header: t("modelAdd.credentialsTableColumns.provider"),
      size: 320,
      enableSorting: false,
      cell: ({ row }) => (
        <CredentialProviderCell
          provider={row.original.credential_info?.custom_llm_provider}
          federated={inferAuthMethod(row.original.credential_values) === "federation"}
        />
      ),
    },
    {
      id: "source",
      accessorFn: (row) => row.source ?? "db",
      meta: { title: t("modelAdd.credentialsTableColumns.source") },
      header: t("modelAdd.credentialsTableColumns.source"),
      size: 140,
      enableSorting: false,
      cell: ({ row }) =>
        row.original.source === "config" ? (
          <StatusBadge tone="neutral" label={t("modelAdd.credentialsTableColumns.configSource")} />
        ) : (
          <StatusBadge tone="info" label="DB" />
        ),
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
      header: () => <span className="sr-only">{t("modelAdd.credentialsTableColumns.actions")}</span>,
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
