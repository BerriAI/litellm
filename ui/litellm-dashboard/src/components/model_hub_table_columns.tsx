import { ColumnDef } from "@tanstack/react-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { Copy, Info } from "lucide-react";
import { cn } from "@/lib/utils";

interface ModelHubData {
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
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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

// Categorical capability-color palette. Each index cycles through a set of
// tags; added to the raw-colors eslintrc override since these are
// categorical by design.
const capabilityColors = [
  "bg-emerald-100 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300",
  "bg-blue-100 text-blue-800 dark:bg-blue-950/40 dark:text-blue-300",
  "bg-purple-100 text-purple-800 dark:bg-purple-950/40 dark:text-purple-300",
  "bg-orange-100 text-orange-800 dark:bg-orange-950/40 dark:text-orange-300",
  "bg-red-100 text-red-800 dark:bg-red-950/40 dark:text-red-300",
  "bg-amber-100 text-amber-800 dark:bg-amber-950/40 dark:text-amber-300",
];

function CopyIconButton({
  value,
  onCopy,
  title,
}: {
  value: string;
  onCopy: (text: string) => void;
  title: string;
}) {
  return (
    <TooltipProvider>
      <Tooltip>
        <TooltipTrigger asChild>
          <button
            type="button"
            onClick={() => onCopy(value)}
            className="cursor-pointer text-muted-foreground hover:text-primary"
            aria-label={title}
          >
            <Copy size={12} />
          </button>
        </TooltipTrigger>
        <TooltipContent>{title}</TooltipContent>
      </Tooltip>
    </TooltipProvider>
  );
}

export const modelHubColumns = (
  showModal: (model: ModelHubData) => void,
  copyToClipboard: (text: string) => void,
  publicPage: boolean = false,
): ColumnDef<ModelHubData>[] => {
  const allColumns: ColumnDef<ModelHubData>[] = [
    {
      header: "Public Model Name",
      accessorKey: "model_group",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const model = row.original;
        return (
          <div className="space-y-1">
            <div className="flex items-center space-x-2">
              <span className="font-medium text-sm">{model.model_group}</span>
              <CopyIconButton
                value={model.model_group}
                onCopy={copyToClipboard}
                title="Copy model name"
              />
            </div>
            <div className="md:hidden">
              <span className="text-xs text-muted-foreground">
                {model.providers.join(", ")}
              </span>
            </div>
          </div>
        );
      },
    },
    {
      header: "Provider",
      accessorKey: "providers",
      enableSorting: true,
      sortingFn: (rowA, rowB) =>
        rowA.original.providers.join(", ").localeCompare(
          rowB.original.providers.join(", "),
        ),
      cell: ({ row }) => {
        const model = row.original;
        return (
          <div className="flex flex-wrap gap-1">
            {model.providers.slice(0, 2).map((provider) => (
              <Badge
                key={provider}
                variant="outline"
                className="text-xs bg-blue-100 text-blue-800 dark:bg-blue-950/40 dark:text-blue-300 border-blue-200 dark:border-blue-800"
              >
                {provider}
              </Badge>
            ))}
            {model.providers.length > 2 && (
              <span className="text-xs text-muted-foreground">
                +{model.providers.length - 2}
              </span>
            )}
          </div>
        );
      },
      meta: { className: "hidden md:table-cell" },
    },
    {
      header: "Mode",
      accessorKey: "mode",
      enableSorting: true,
      sortingFn: "alphanumeric",
      cell: ({ row }) => {
        const model = row.original;
        return model.mode ? (
          <Badge
            variant="outline"
            className="text-xs bg-emerald-100 text-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800"
          >
            {model.mode}
          </Badge>
        ) : (
          <span className="text-muted-foreground">-</span>
        );
      },
      meta: { className: "hidden lg:table-cell" },
    },
    {
      header: "Tokens",
      accessorKey: "max_input_tokens",
      enableSorting: true,
      sortingFn: (rowA, rowB) => {
        const tokensA =
          (rowA.original.max_input_tokens || 0) +
          (rowA.original.max_output_tokens || 0);
        const tokensB =
          (rowB.original.max_input_tokens || 0) +
          (rowB.original.max_output_tokens || 0);
        return tokensA - tokensB;
      },
      cell: ({ row }) => {
        const model = row.original;
        return (
          <div className="space-y-1">
            <span className="text-xs">
              {model.max_input_tokens
                ? formatTokens(model.max_input_tokens)
                : "-"}{" "}
              /{" "}
              {model.max_output_tokens
                ? formatTokens(model.max_output_tokens)
                : "-"}
            </span>
          </div>
        );
      },
      meta: { className: "hidden lg:table-cell" },
    },
    {
      header: "Cost/1M",
      accessorKey: "input_cost_per_token",
      enableSorting: true,
      sortingFn: (rowA, rowB) => {
        const costA =
          (rowA.original.input_cost_per_token || 0) +
          (rowA.original.output_cost_per_token || 0);
        const costB =
          (rowB.original.input_cost_per_token || 0) +
          (rowB.original.output_cost_per_token || 0);
        return costA - costB;
      },
      cell: ({ row }) => {
        const model = row.original;
        return (
          <div className="space-y-1">
            <span className="text-xs block">
              {model.input_cost_per_token
                ? formatCost(model.input_cost_per_token)
                : "-"}
            </span>
            <span className="text-xs text-muted-foreground block">
              {model.output_cost_per_token
                ? formatCost(model.output_cost_per_token)
                : "-"}
            </span>
          </div>
        );
      },
    },
    {
      header: "Features",
      accessorKey: "capabilities",
      enableSorting: false,
      cell: ({ row }) => {
        const model = row.original;
        const capabilities = getModelCapabilities(model);
        return (
          <div className="flex flex-wrap gap-1">
            {capabilities.length === 0 ? (
              <span className="text-muted-foreground text-xs">-</span>
            ) : (
              capabilities.map((capability, index) => (
                <Badge
                  key={capability}
                  variant="outline"
                  className={cn(
                    "text-xs",
                    capabilityColors[index % capabilityColors.length],
                  )}
                >
                  {formatCapabilityName(capability)}
                </Badge>
              ))
            )}
          </div>
        );
      },
    },
    {
      header: "Public",
      accessorKey: "is_public_model_group",
      enableSorting: true,
      sortingFn: (rowA, rowB) => {
        const publicA =
          rowA.original.is_public_model_group === true ? 1 : 0;
        const publicB =
          rowB.original.is_public_model_group === true ? 1 : 0;
        return publicA - publicB;
      },
      cell: ({ row }) => {
        const model = row.original;
        return model.is_public_model_group === true ? (
          <Badge variant="default" className="text-xs">
            Yes
          </Badge>
        ) : (
          <Badge variant="outline" className="text-xs">
            No
          </Badge>
        );
      },
      meta: { className: "hidden md:table-cell" },
    },
    {
      header: "Details",
      id: "details",
      enableSorting: false,
      cell: ({ row }) => (
        <Button
          size="sm"
          variant="secondary"
          onClick={() => showModal(row.original)}
        >
          <Info className="h-3 w-3" />
          <span className="hidden lg:inline">Details</span>
          <span className="lg:hidden">Info</span>
        </Button>
      ),
    },
  ];

  if (publicPage) {
    return allColumns.filter((column) => {
      if (
        "accessorKey" in column &&
        column.accessorKey === "is_public_model_group"
      ) {
        return false;
      }
      return true;
    });
  }

  return allColumns;
};
