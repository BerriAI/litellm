"use client";

import { ColumnDef } from "@tanstack/react-table";
import { Copy, Info, MoreHorizontal } from "lucide-react";

import { DataTableSortHeader } from "@/components/shared/DataTable";
import { IdentityCell, StatusBadge } from "@/components/shared/table_cells";
import { Badge } from "@/components/ui/badge";
import { buttonVariants } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { cn } from "@/lib/cva.config";
import { copyToClipboard } from "@/utils/dataUtils";
import { getProviderLogoAndName } from "@/components/provider_info_helpers";

export interface ModelHubData {
  model_group: string;
  providers: string[];
  max_input_tokens?: number;
  max_output_tokens?: number;
  input_cost_per_token?: number;
  output_cost_per_token?: number;
  mode?: string;
  tpm?: number;
  rpm?: number;
  supports_parallel_function_calling: boolean;
  supports_vision: boolean;
  supports_function_calling: boolean;
  supported_openai_params?: string[];
  is_public_model_group: boolean;
  [key: string]: any;
}

const formatCapabilityName = (key: string) =>
  key
    .replace(/^supports_/, "")
    .split("_")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");

const getModelCapabilities = (model: ModelHubData) =>
  Object.entries(model)
    .filter(([key, value]) => key.startsWith("supports_") && value === true)
    .map(([key]) => key);

const formatCost = (cost: number) => `$${(cost * 1_000_000).toFixed(2)}`;

const formatTokens = (tokens: number) => {
  if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1)}M`;
  if (tokens >= 1_000) return `${(tokens / 1_000).toFixed(1)}K`;
  return tokens.toString();
};

interface ModelHubRowActionsProps {
  model: ModelHubData;
  onModelClick: (model: ModelHubData) => void;
}

function ModelHubRowActions({ model, onModelClick }: ModelHubRowActionsProps) {
  return (
    <DropdownMenu>
      <DropdownMenuTrigger
        aria-label="Open model actions"
        data-testid={`model-hub-actions-${model.model_group}`}
        className={cn(buttonVariants({ variant: "ghost", size: "icon-sm" }), "text-muted-foreground")}
      >
        <MoreHorizontal className="size-4" />
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-52">
        <DropdownMenuItem data-testid="model-hub-action-details" onClick={() => onModelClick(model)}>
          <Info />
          View details
        </DropdownMenuItem>
        <DropdownMenuItem
          data-testid="model-hub-action-copy"
          onClick={() => void copyToClipboard(model.model_group, "Model name copied")}
        >
          <Copy />
          Copy model name
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}

interface ModelHubTableColumnsDeps {
  onModelClick: (model: ModelHubData) => void;
}

export const getModelHubTableColumns = ({ onModelClick }: ModelHubTableColumnsDeps): ColumnDef<ModelHubData>[] => [
  {
    id: "model_group",
    accessorKey: "model_group",
    meta: { title: "Public Model Name" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Public Model Name" />,
    size: 220,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => (
      <IdentityCell title={row.original.model_group} className="max-w-72" onClick={() => onModelClick(row.original)} />
    ),
  },
  {
    id: "providers",
    accessorKey: "providers",
    meta: { title: "Provider", skeleton: "chips", className: "hidden md:table-cell" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Provider" />,
    size: 150,
    enableSorting: true,
    sortingFn: (rowA, rowB) => {
      const namesA = rowA.original.providers.map((p) => getProviderLogoAndName(p).displayName).join(", ");
      const namesB = rowB.original.providers.map((p) => getProviderLogoAndName(p).displayName).join(", ");
      return namesA.localeCompare(namesB);
    },
    cell: ({ row }) => {
      const providers = row.original.providers;
      return (
        <div className="flex flex-wrap gap-1">
          {providers.slice(0, 2).map((provider) => (
            <Badge key={provider} variant="secondary">
              {getProviderLogoAndName(provider).displayName}
            </Badge>
          ))}
          {providers.length > 2 && <span className="text-xs text-muted-foreground">+{providers.length - 2}</span>}
        </div>
      );
    },
  },
  {
    id: "mode",
    accessorKey: "mode",
    meta: { title: "Mode", className: "hidden lg:table-cell" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Mode" />,
    size: 110,
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) =>
      row.original.mode ? (
        <span className="font-mono text-xs text-muted-foreground">{row.original.mode}</span>
      ) : (
        <span className="text-xs text-muted-foreground">-</span>
      ),
  },
  {
    id: "max_input_tokens",
    accessorKey: "max_input_tokens",
    meta: { title: "Tokens", className: "hidden lg:table-cell" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Tokens" />,
    size: 110,
    enableSorting: true,
    sortingFn: (rowA, rowB) => {
      const tokensA = (rowA.original.max_input_tokens || 0) + (rowA.original.max_output_tokens || 0);
      const tokensB = (rowB.original.max_input_tokens || 0) + (rowB.original.max_output_tokens || 0);
      return tokensA - tokensB;
    },
    cell: ({ row }) => {
      const model = row.original;
      return (
        <span className="text-xs tabular-nums">
          {model.max_input_tokens ? formatTokens(model.max_input_tokens) : "-"} /{" "}
          {model.max_output_tokens ? formatTokens(model.max_output_tokens) : "-"}
        </span>
      );
    },
  },
  {
    id: "input_cost_per_token",
    accessorKey: "input_cost_per_token",
    meta: { title: "Cost/1M", skeleton: "twoLine" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Cost/1M" />,
    size: 110,
    enableSorting: true,
    sortingFn: (rowA, rowB) => {
      const costA = (rowA.original.input_cost_per_token || 0) + (rowA.original.output_cost_per_token || 0);
      const costB = (rowB.original.input_cost_per_token || 0) + (rowB.original.output_cost_per_token || 0);
      return costA - costB;
    },
    cell: ({ row }) => {
      const model = row.original;
      return (
        <div className="flex flex-col gap-0.5 text-xs tabular-nums">
          <span>{model.input_cost_per_token ? formatCost(model.input_cost_per_token) : "-"}</span>
          <span className="text-muted-foreground">
            {model.output_cost_per_token ? formatCost(model.output_cost_per_token) : "-"}
          </span>
        </div>
      );
    },
  },
  {
    id: "capabilities",
    meta: { title: "Features", skeleton: "chips" },
    header: "Features",
    size: 220,
    enableSorting: false,
    cell: ({ row }) => {
      const capabilities = getModelCapabilities(row.original);
      if (capabilities.length === 0) {
        return <span className="text-xs text-muted-foreground">-</span>;
      }
      return (
        <div className="flex flex-wrap gap-1">
          {capabilities.map((capability) => (
            <Badge key={capability} variant="outline">
              {formatCapabilityName(capability)}
            </Badge>
          ))}
        </div>
      );
    },
  },
  {
    id: "is_public_model_group",
    accessorKey: "is_public_model_group",
    meta: { title: "Public", skeleton: "badge", className: "hidden md:table-cell" },
    header: ({ column }) => <DataTableSortHeader column={column} title="Public" />,
    size: 100,
    enableSorting: true,
    sortingFn: (rowA, rowB) => {
      const publicA = rowA.original.is_public_model_group === true ? 1 : 0;
      const publicB = rowB.original.is_public_model_group === true ? 1 : 0;
      return publicA - publicB;
    },
    cell: ({ row }) =>
      row.original.is_public_model_group === true ? (
        <StatusBadge tone="success" label="Yes" />
      ) : (
        <StatusBadge tone="neutral" label="No" />
      ),
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
        <ModelHubRowActions model={row.original} onModelClick={onModelClick} />
      </div>
    ),
  },
];
