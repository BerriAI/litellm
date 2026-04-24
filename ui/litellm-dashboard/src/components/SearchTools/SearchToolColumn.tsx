import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { ColumnDef } from "@tanstack/react-table";
import TableIconActionButton from "../common_components/IconActionButton/TableIconActionButtons/TableIconActionButton";
import { SearchTool } from "./types";

export const searchToolColumns = (
  onView: (searchToolId: string) => void,
  onEdit: (searchToolId: string) => void,
  onDelete: (searchToolId: string) => void,
  availableProviders: Array<{ provider_name: string; ui_friendly_name: string }>,
): ColumnDef<SearchTool>[] => [
  {
    header: "Search Tool ID",
    accessorKey: "search_tool_id",
    cell: ({ row }) => {
      const tool = row.original;
      const isFromConfig = tool.is_from_config;

      if (isFromConfig) {
        return <span className="text-xs">-</span>;
      }

      return (
        <button
          type="button"
          onClick={() => onView(tool.search_tool_id!)}
          className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/30 dark:hover:bg-blue-950/60 text-xs font-normal px-2 py-0.5 text-left cursor-pointer max-w-40 rounded"
        >
          <span className="truncate block">{tool.search_tool_id}</span>
        </button>
      );
    },
  },
  {
    header: "Name",
    accessorKey: "search_tool_name",
    cell: (info) => <span className="font-medium">{info.getValue() as string}</span>,
  },
  {
    header: "Provider",
    id: "provider",
    cell: ({ row }) => {
      const tool = row.original;
      const provider = tool.litellm_params.search_provider;
      const providerInfo = availableProviders.find((p) => p.provider_name === provider);
      const displayName = providerInfo?.ui_friendly_name || provider;
      return <span className="text-sm">{displayName}</span>;
    },
  },
  {
    header: "Created At",
    accessorKey: "created_at",
    cell: ({ row }) => (
      <span className="text-xs">
        {row.original.created_at ? new Date(row.original.created_at).toLocaleDateString() : "-"}
      </span>
    ),
  },
  {
    header: "Updated At",
    accessorKey: "updated_at",
    cell: ({ row }) => (
      <span className="text-xs">
        {row.original.updated_at ? new Date(row.original.updated_at).toLocaleDateString() : "-"}
      </span>
    ),
  },
  {
    header: "Source",
    id: "source",
    cell: ({ row }) => {
      const isFromConfig = row.original.is_from_config ?? false;
      return (
        <Badge
          className={cn(
            "text-xs",
            isFromConfig
              ? "bg-muted text-muted-foreground"
              : "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
          )}
        >
          {isFromConfig ? "Config" : "DB"}
        </Badge>
      );
    },
  },
  {
    header: "Actions",
    id: "actions",
    cell: ({ row }) => {
      const tool = row.original;
      const toolId = tool.search_tool_id;
      const isFromConfig = tool.is_from_config ?? false;

      return (
        <div className="flex items-center gap-2">
          <TableIconActionButton
            variant="Edit"
            tooltipText="Edit search tool"
            disabled={isFromConfig}
            disabledTooltipText="Config search tool cannot be edited on the dashboard. Please edit it from the config file."
            onClick={() => {
              if (toolId && !isFromConfig) {
                onEdit(toolId);
              }
            }}
          />
          <TableIconActionButton
            variant="Delete"
            tooltipText="Delete search tool"
            disabled={isFromConfig}
            disabledTooltipText="Config search tool cannot be deleted on the dashboard. Please delete it from the config file."
            onClick={() => {
              if (toolId && !isFromConfig) {
                onDelete(toolId);
              }
            }}
          />
        </div>
      );
    },
  },
];
