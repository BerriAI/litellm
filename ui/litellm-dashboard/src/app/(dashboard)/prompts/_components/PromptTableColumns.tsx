"use client";

import { ColumnDef } from "@tanstack/react-table";
import { Copy, MoreHorizontal, Trash2 } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { CellTooltip, DateCell, IdentityCell, StatusBadge, StatusTone } from "@/components/shared/table_cells";
import { PromptSpec } from "@/components/networking";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";
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

import { extractModel, getProviderFromModelHub, ModelGroupInfo } from "./prompt_utils";

const ENVIRONMENT_TONE: Record<string, StatusTone> = {
  production: "error",
  staging: "warning",
  development: "success",
};

function PromptModelCell({ prompt, modelHubData }: { prompt: PromptSpec; modelHubData: Map<string, ModelGroupInfo> }) {
  const model = extractModel(prompt);
  if (!model) {
    return <span className="text-sm text-muted-foreground">-</span>;
  }

  const provider = getProviderFromModelHub(model, modelHubData);
  const { logo } = provider ? getProviderLogoAndName(provider) : { logo: "" };

  return (
    <CellTooltip
      content={model}
      trigger={
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
          ) : (
            <span className="flex size-4 shrink-0 items-center justify-center rounded-full bg-muted text-xs text-muted-foreground">
              {provider?.charAt(0) || "-"}
            </span>
          )}
          <span className="max-w-40 truncate text-sm">{model}</span>
        </div>
      }
    />
  );
}

interface PromptRowActionsProps {
  prompt: PromptSpec;
  isAdmin: boolean;
  onDeleteClick?: (id: string, name: string) => void;
}

function PromptRowActions({ prompt, isAdmin, onDeleteClick }: PromptRowActionsProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label="Open prompt actions"
        data-testid={`prompt-actions-${prompt.prompt_id}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem
          data-testid="prompt-action-copy"
          onClick={() => void copyToClipboard(prompt.prompt_id, "Prompt ID copied")}
        >
          <Copy />
          Copy prompt ID
        </DropdownMenuItem>
        {isAdmin && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem
              variant="destructive"
              data-testid="prompt-action-delete"
              onClick={() => onDeleteClick?.(prompt.prompt_id, prompt.prompt_id || "Unknown Prompt")}
            >
              <Trash2 />
              Delete
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface PromptTableColumnsDeps {
  modelHubData: Map<string, ModelGroupInfo>;
  isAdmin: boolean;
  onPromptClick?: (id: string) => void;
  onDeleteClick?: (id: string, name: string) => void;
}

export const getPromptTableColumns = ({
  modelHubData,
  isAdmin,
  onPromptClick,
  onDeleteClick,
}: PromptTableColumnsDeps): ColumnDef<PromptSpec>[] => [
  {
    id: "prompt_id",
    accessorKey: "prompt_id",
    meta: { title: "Prompt ID" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Prompt ID" />,
    size: 220,
    enableSorting: true,
    cell: ({ row }) => (
      <IdentityCell
        title={row.original.prompt_id}
        titleClassName="font-mono text-xs font-normal"
        className="max-w-60"
        onClick={onPromptClick ? () => onPromptClick(row.original.prompt_id) : undefined}
      />
    ),
  },
  {
    id: "model",
    meta: { title: "Model" },
    header: "Model",
    size: 200,
    enableSorting: false,
    cell: ({ row }) => <PromptModelCell prompt={row.original} modelHubData={modelHubData} />,
  },
  {
    id: "created_at",
    accessorKey: "created_at",
    sortingFn: "datetime",
    meta: { title: "Created At" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Created At" />,
    size: 160,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.created_at} />,
  },
  {
    id: "updated_at",
    accessorKey: "updated_at",
    sortingFn: "datetime",
    meta: { title: "Updated At" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Updated At" />,
    size: 160,
    enableSorting: true,
    cell: ({ row }) => <DateCell value={row.original.updated_at} />,
  },
  {
    id: "environment",
    accessorKey: "environment",
    meta: { title: "Environment", skeleton: "badge" },
    header: "Environment",
    size: 130,
    enableSorting: false,
    cell: ({ row }) => {
      const environment = row.original.environment || "development";
      return <StatusBadge tone={ENVIRONMENT_TONE[environment] ?? "neutral"} label={environment} />;
    },
  },
  {
    id: "created_by",
    accessorKey: "created_by",
    meta: { title: "Created By" },
    header: "Created By",
    size: 160,
    enableSorting: false,
    cell: ({ row }) => {
      const createdBy = row.original.created_by;
      return (
        <span className="block max-w-60 truncate text-sm text-muted-foreground" title={createdBy}>
          {createdBy || "-"}
        </span>
      );
    },
  },
  {
    id: "prompt_type",
    accessorKey: "prompt_info.prompt_type",
    meta: { title: "Type" },
    header: "Type",
    size: 140,
    enableSorting: false,
    cell: ({ row }) => {
      const promptType = row.original.prompt_info.prompt_type;
      return (
        <span className="block max-w-40 truncate text-sm" title={promptType}>
          {promptType}
        </span>
      );
    },
  },
  {
    id: "actions",
    meta: { className: "text-right", headerClassName: "text-right" },
    header: () => <span className="sr-only">Actions</span>,
    size: 64,
    enableSorting: false,
    enableHiding: false,
    cell: ({ row }) => (
      <div className="flex justify-end">
        <PromptRowActions prompt={row.original} isAdmin={isAdmin} onDeleteClick={onDeleteClick} />
      </div>
    ),
  },
];
