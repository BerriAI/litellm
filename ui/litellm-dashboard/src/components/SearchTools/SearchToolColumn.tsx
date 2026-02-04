import { ColumnDef } from "@tanstack/react-table";
import { SearchTool } from "./types";
import { Icon } from "@tremor/react";
import { PencilAltIcon, TrashIcon } from "@heroicons/react/outline";

export const searchToolColumns = (
  onView: (searchToolId: string) => void,
  onEdit: (searchToolId: string) => void,
  onDelete: (searchToolId: string) => void,
  availableProviders: Array<{ provider_name: string; ui_friendly_name: string }>,
): ColumnDef<SearchTool>[] => [
  {
    accessorKey: "search_tool_id",
    header: "Search Tool ID",
    cell: ({ row }) => (
      <button
        onClick={() => onView(row.original.search_tool_id!)}
        className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left w-full truncate whitespace-nowrap cursor-pointer max-w-[15ch]"
      >
        {row.original.search_tool_id?.slice(0, 7)}...
      </button>
    ),
  },
  {
    accessorKey: "search_tool_name",
    header: "Name",
    cell: ({ getValue }) => <span className="font-medium">{getValue() as string}</span>,
  },
  {
    id: "provider",
    header: "Provider",
    cell: ({ row }) => {
      const provider = row.original.litellm_params.search_provider;
      const providerInfo = availableProviders.find((p) => p.provider_name === provider);
      const displayName = providerInfo?.ui_friendly_name || provider;

      return <span className="text-sm">{displayName}</span>;
    },
  },
  {
    header: "Created At",
    accessorKey: "created_at",
    sortingFn: "datetime",
    cell: ({ row }) => {
      const tool = row.original;
      return <span className="text-xs">{tool.created_at ? new Date(tool.created_at).toLocaleDateString() : "-"}</span>;
    },
  },
  {
    header: "Updated At",
    accessorKey: "updated_at",
    sortingFn: "datetime",
    cell: ({ row }) => {
      const tool = row.original;
      return <span className="text-xs">{tool.updated_at ? new Date(tool.updated_at).toLocaleDateString() : "-"}</span>;
    },
  },
  {
    id: "actions",
    header: "Actions",
    cell: ({ row }) => (
      <div className="flex items-center gap-2">
        <Icon
          icon={PencilAltIcon}
          size="sm"
          onClick={() => onEdit(row.original.search_tool_id!)}
          className="cursor-pointer"
        />
        <Icon
          icon={TrashIcon}
          size="sm"
          onClick={() => onDelete(row.original.search_tool_id!)}
          className="cursor-pointer"
        />
      </div>
    ),
  },
];
