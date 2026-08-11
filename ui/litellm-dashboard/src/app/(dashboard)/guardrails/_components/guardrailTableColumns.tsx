"use client";

import { ColumnDef } from "@tanstack/react-table";
import { MoreHorizontal, Trash2 } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdentityCell, StatusBadge } from "@/components/shared/table_cells";
import { Guardrail, GuardrailDefinitionLocation } from "@/components/guardrails/types";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";

import { getGuardrailLogoAndName } from "./guardrail_info_helpers";
import { Logo } from "@/components/molecules/logo/Logo";
import { useTranslation } from "react-i18next";

function GuardrailProviderCell({ provider }: { provider: string }) {
  const { logo, displayName } = getGuardrailLogoAndName(provider);
  return (
    <div className="flex items-center gap-2">
      <Logo src={logo} label={displayName} className="size-4 shrink-0" />
      <span className="truncate text-sm">{displayName}</span>
    </div>
  );
}

interface GuardrailRowActionsProps {
  guardrail: Guardrail;
  onDeleteClick: (guardrailId: string, guardrailName: string) => void;
}

function GuardrailRowActions({ guardrail, onDeleteClick }: GuardrailRowActionsProps) {
  const { t } = useTranslation("gateway");
  const isConfigGuardrail = guardrail.guardrail_definition_location === GuardrailDefinitionLocation.CONFIG;

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("guardrailsPage.actions.open")}
        data-testid={`guardrail-actions-${guardrail.guardrail_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          variant="destructive"
          disabled={isConfigGuardrail}
          data-testid="guardrail-action-delete"
          title={isConfigGuardrail ? t("guardrailsPage.actions.configDeleteHint") : undefined}
          onClick={() =>
            onDeleteClick(guardrail.guardrail_id, guardrail.guardrail_name || t("guardrailsPage.values.unnamed"))
          }
        >
          <Trash2 />
          {t("guardrailsPage.actions.delete")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface GuardrailTableColumnsDeps {
  onGuardrailClick: (guardrailId: string) => void;
  onDeleteClick: (guardrailId: string, guardrailName: string) => void;
  t: (key: string) => string;
}

export const getGuardrailTableColumns = ({
  onGuardrailClick,
  onDeleteClick,
  t,
}: GuardrailTableColumnsDeps): ColumnDef<Guardrail>[] => [
  {
    id: "guardrail_id",
    accessorKey: "guardrail_id",
    meta: { title: t("guardrailsPage.columns.id") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("guardrailsPage.columns.id")} />,
    size: 200,
    enableSorting: true,
    cell: ({ row }) => (
      <IdentityCell
        title={row.original.guardrail_id}
        titleClassName="font-mono text-xs font-normal"
        onClick={() => onGuardrailClick(row.original.guardrail_id)}
      />
    ),
  },
  {
    id: "guardrail_name",
    accessorKey: "guardrail_name",
    meta: { title: t("guardrailsPage.columns.name") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("guardrailsPage.columns.name")} />,
    size: 200,
    enableSorting: true,
    cell: ({ row }) => {
      const name = row.original.guardrail_name;
      return (
        <span className="block truncate text-sm font-medium" title={name ?? undefined}>
          {name || "-"}
        </span>
      );
    },
  },
  {
    id: "provider",
    meta: { title: t("guardrailsPage.columns.provider") },
    header: t("guardrailsPage.columns.provider"),
    size: 180,
    enableSorting: false,
    cell: ({ row }) => <GuardrailProviderCell provider={row.original.litellm_params.guardrail} />,
  },
  {
    id: "mode",
    meta: { title: t("guardrailsPage.columns.mode") },
    header: t("guardrailsPage.columns.mode"),
    size: 130,
    enableSorting: false,
    cell: ({ row }) => (
      <span className="font-mono text-xs text-muted-foreground">{row.original.litellm_params.mode}</span>
    ),
  },
  {
    id: "default_on",
    meta: { title: t("guardrailsPage.columns.defaultOn") },
    header: t("guardrailsPage.columns.defaultOn"),
    size: 120,
    enableSorting: false,
    cell: ({ row }) => {
      const isDefaultOn = !!row.original.litellm_params?.default_on;
      return (
        <StatusBadge
          tone={isDefaultOn ? "success" : "neutral"}
          label={isDefaultOn ? t("guardrailsPage.values.defaultOn") : t("guardrailsPage.values.defaultOff")}
        />
      );
    },
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    meta: { title: t("guardrailsPage.columns.createdAt") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("guardrailsPage.columns.createdAt")} />,
    size: 150,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} />,
  },
  {
    id: "updated_at",
    accessorKey: "updated_at",
    meta: { title: t("guardrailsPage.columns.updatedAt") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("guardrailsPage.columns.updatedAt")} />,
    size: 150,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.updated_at} />,
  },
  {
    id: "actions",
    meta: { className: "text-right", headerClassName: "text-right" },
    header: () => <span className="sr-only">{t("guardrailsPage.columns.actions")}</span>,
    size: 64,
    enableSorting: false,
    enableHiding: false,
    cell: ({ row }) => (
      <div className="flex justify-end">
        <GuardrailRowActions guardrail={row.original} onDeleteClick={onDeleteClick} />
      </div>
    ),
  },
];
