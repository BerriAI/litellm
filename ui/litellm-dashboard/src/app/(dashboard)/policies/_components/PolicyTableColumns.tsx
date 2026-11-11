"use client";

import { ColumnDef } from "@tanstack/react-table";
import type { ParseKeys, TFunction } from "i18next";
import { MoreHorizontal, Pencil, Trash2 } from "lucide-react";
import { useTranslation } from "react-i18next";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { DateCell, IdentityCell, StatusBadge } from "@/components/shared/table_cells";
import { Policy } from "@/components/policies/types";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";

export interface PolicyRow {
  policy_name: string;
  primaryPolicy: Policy;
  versionCount: number;
}

const CONFIG_POLICY_HINT_KEY: ParseKeys = "policies.policyTable.configHint";

function GuardrailChips({ guardrails, tone }: { guardrails: string[]; tone: "success" | "error" }) {
  if (guardrails.length === 0) {
    return <span className="text-muted-foreground">-</span>;
  }
  return (
    <div className="flex flex-wrap items-center gap-1">
      {guardrails.slice(0, 2).map((guardrail) => (
        <StatusBadge key={guardrail} tone={tone} label={guardrail} />
      ))}
      {guardrails.length > 2 && (
        <StatusBadge tone="neutral" label={`+${guardrails.length - 2}`} tooltip={guardrails.slice(2).join(", ")} />
      )}
    </div>
  );
}

interface PolicyRowActionsProps {
  policy: Policy;
  onEditClick: (policy: Policy) => void;
  onDeleteClick: (policyId: string, policyName: string) => void;
}

function PolicyRowActions({ policy, onEditClick, onDeleteClick }: PolicyRowActionsProps) {
  const { t } = useTranslation();
  const isConfigPolicy = policy.definition_location === "config";

  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label={t("policies.policyTable.actions.open")}
        data-testid={`policy-actions-${policy.policy_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          data-testid="policy-action-edit"
          disabled={isConfigPolicy}
          title={isConfigPolicy ? t(CONFIG_POLICY_HINT_KEY) : undefined}
          onClick={() => onEditClick(policy)}
        >
          <Pencil />
          {t("policies.policyTable.actions.edit")}
        </DropdownMenuItem>
        <DropdownMenuSeparator />
        <DropdownMenuItem
          variant="destructive"
          data-testid="policy-action-delete"
          disabled={isConfigPolicy}
          title={isConfigPolicy ? t(CONFIG_POLICY_HINT_KEY) : undefined}
          onClick={() => onDeleteClick(policy.policy_id, policy.policy_name || t("policies.policyTable.unnamedPolicy"))}
        >
          <Trash2 />
          {t("policies.policyTable.actions.delete")}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface PolicyTableColumnsDeps {
  isAdmin: boolean;
  onViewClick: (policyId: string) => void;
  onEditClick: (policy: Policy) => void;
  onDeleteClick: (policyId: string, policyName: string) => void;
  t: TFunction;
}

export const getPolicyTableColumns = ({
  isAdmin,
  onViewClick,
  onEditClick,
  onDeleteClick,
  t,
}: PolicyTableColumnsDeps): ColumnDef<PolicyRow>[] => [
  {
    id: "policy_name",
    accessorKey: "policy_name",
    meta: { title: t("policies.policyTable.columns.name"), skeleton: "twoLine" },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("policies.policyTable.columns.name")} />,
    size: 220,
    enableSorting: true,
    cell: ({ row }) => {
      const isConfigPolicy = row.original.primaryPolicy.definition_location === "config";
      const versionBadge =
        row.original.versionCount > 1 ? (
          <StatusBadge
            tone="neutral"
            label={t("policies.policyTable.versionCount", { count: row.original.versionCount })}
          />
        ) : undefined;
      return (
        <IdentityCell
          title={row.original.policy_name}
          titleClassName="max-w-60"
          badge={
            isConfigPolicy ? (
              <StatusBadge
                tone="neutral"
                label={t("policies.policyTable.configBadge")}
                tooltip={t(CONFIG_POLICY_HINT_KEY)}
              />
            ) : (
              versionBadge
            )
          }
          onClick={isConfigPolicy ? undefined : () => onViewClick(row.original.primaryPolicy.policy_id)}
        />
      );
    },
  },
  {
    id: "description",
    accessorFn: (row) => row.primaryPolicy.description ?? "",
    meta: { title: t("policies.policyTable.columns.description") },
    header: t("policies.policyTable.columns.description"),
    size: 220,
    enableSorting: false,
    cell: ({ row }) => {
      const description = row.original.primaryPolicy.description;
      if (!description) {
        return <span className="text-muted-foreground">-</span>;
      }
      return (
        <span className="block max-w-60 truncate text-muted-foreground" title={description}>
          {description}
        </span>
      );
    },
  },
  {
    id: "inherit",
    accessorFn: (row) => row.primaryPolicy.inherit ?? "",
    meta: { title: t("policies.policyTable.columns.inherit"), skeleton: "badge" },
    header: t("policies.policyTable.columns.inherit"),
    size: 150,
    enableSorting: false,
    cell: ({ row }) => {
      const inherit = row.original.primaryPolicy.inherit;
      if (!inherit) {
        return <span className="text-muted-foreground">-</span>;
      }
      return <StatusBadge tone="info" label={inherit} />;
    },
  },
  {
    id: "guardrails_add",
    meta: { title: t("policies.policyTable.columns.guardrailsAdd"), skeleton: "chips" },
    header: t("policies.policyTable.columns.guardrailsAdd"),
    size: 180,
    enableSorting: false,
    cell: ({ row }) => <GuardrailChips guardrails={row.original.primaryPolicy.guardrails_add ?? []} tone="success" />,
  },
  {
    id: "guardrails_remove",
    meta: { title: t("policies.policyTable.columns.guardrailsRemove"), skeleton: "chips" },
    header: t("policies.policyTable.columns.guardrailsRemove"),
    size: 180,
    enableSorting: false,
    cell: ({ row }) => <GuardrailChips guardrails={row.original.primaryPolicy.guardrails_remove ?? []} tone="error" />,
  },
  {
    id: "model_condition",
    meta: { title: t("policies.policyTable.columns.modelCondition") },
    header: t("policies.policyTable.columns.modelCondition"),
    size: 160,
    enableSorting: false,
    cell: ({ row }) => {
      const model = row.original.primaryPolicy.condition?.model;
      if (!model) {
        return <span className="text-muted-foreground">-</span>;
      }
      return (
        <code className="block max-w-40 truncate rounded-sm bg-muted px-1 py-0.5 font-mono text-xs" title={model}>
          {model}
        </code>
      );
    },
  },
  {
    id: "created_at",
    accessorFn: (row) => row.primaryPolicy.created_at ?? "",
    meta: { title: t("policies.policyTable.columns.createdAt") },
    header: ({ column }) => <DataTableSortHeader column={column} title={t("policies.policyTable.columns.createdAt")} />,
    size: 150,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.primaryPolicy.created_at} />,
  },
  ...(isAdmin
    ? [
        {
          id: "actions",
          meta: { className: "text-right", headerClassName: "text-right" },
          header: () => <span className="sr-only">{t("policies.policyTable.columns.actions")}</span>,
          size: 64,
          enableSorting: false,
          enableHiding: false,
          cell: ({ row }) => (
            <div className="flex justify-end">
              <PolicyRowActions
                policy={row.original.primaryPolicy}
                onEditClick={onEditClick}
                onDeleteClick={onDeleteClick}
              />
            </div>
          ),
        } satisfies ColumnDef<PolicyRow>,
      ]
    : []),
];
