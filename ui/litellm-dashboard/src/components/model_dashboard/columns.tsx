import { ColumnDef } from "@tanstack/react-table";
import { Button, Badge } from "@tremor/react";
import { Tooltip } from "antd";
import { getProviderLogoAndName } from "../provider_info_helpers";
import { ModelData } from "./types";
import { TrashIcon, PencilIcon } from "@heroicons/react/outline";
import DeleteModelButton from "../delete_model_button";

export const columns = (
  premiumUser: boolean,
  setSelectedModelId: (id: string) => void,
  setSelectedTeamId: (id: string) => void,
  getDisplayModelName: (model: any) => string,
  handleEditClick: (model: any) => void,
  handleRefreshClick: () => void,
): ColumnDef<ModelData>[] => [
  {
    header: ({ column }) => (
      <div
        onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        className="flex items-center gap-2 text-xs cursor-pointer text-gray-600 hover:text-gray-900"
      >
        Model ID
        {{
          asc: " ↑",
          desc: " ↓",
        }[column.getIsSorted() as string] ?? " ↕"}
      </div>
    ),
    accessorKey: "model_info.id",
    cell: ({ row }) => {
      const model = row.original;
      return (
        <div className="overflow-hidden">
          <Tooltip title={model.model_info.id}>
            <Button 
              size="xs"
              variant="light"
              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
              onClick={() => setSelectedModelId(model.model_info.id)}
            >
              {model.model_info.id.slice(0, 7)}...
            </Button>
          </Tooltip>
        </div>
      );
    },
  },
  {
    header: ({ column }) => (
      <div
        onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        className="flex items-center gap-2 text-xs cursor-pointer text-gray-600 hover:text-gray-900"
      >
        Model Name
        {{
          asc: " ↑",
          desc: " ↓",
        }[column.getIsSorted() as string] ?? " ↕"}
      </div>
    ),
    accessorKey: "model_name",
    cell: ({ row }) => (
      <p className="text-xs">{getDisplayModelName(row.original) || "-"}</p>
    ),
  },
  {
    header: ({ column }) => (
      <div
        onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        className="flex items-center gap-2 text-xs cursor-pointer text-gray-600 hover:text-gray-900"
      >
        Provider
        {{
          asc: " ↑",
          desc: " ↓",
        }[column.getIsSorted() as string] ?? " ↕"}
      </div>
    ),
    accessorKey: "provider",
    cell: ({ row }) => {
      const model = row.original;
      return (
        <div className="flex items-center space-x-2">
          {model.provider && (
            <img
              src={getProviderLogoAndName(model.provider).logo}
              alt={`${model.provider} logo`}
              className="w-4 h-4"
              onError={(e) => {
                const target = e.target as HTMLImageElement;
                const parent = target.parentElement;
                if (parent) {
                  const fallbackDiv = document.createElement('div');
                  fallbackDiv.className = 'w-4 h-4 rounded-full bg-gray-200 flex items-center justify-center text-xs';
                  fallbackDiv.textContent = model.provider?.charAt(0) || '-';
                  parent.replaceChild(fallbackDiv, target);
                }
              }}
            />
          )}
          <p className="text-xs">{model.provider || "-"}</p>
        </div>
      );
    },
  },
  {
    header: "LiteLLM Model Name",
    cell: ({ row }) => {
      const model = row.original;
      return (
        <Tooltip title={model.litellm_model_name}>
          <pre className="text-xs">
            {model.litellm_model_name
              ? model.litellm_model_name.slice(0, 20) + (model.litellm_model_name.length > 20 ? "..." : "")
              : "-"}
          </pre>
        </Tooltip>
      );
    },
  },
  {
    header: ({ column }) => {
      return (
        <div
          onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
          className="flex items-center gap-2 text-xs cursor-pointer text-gray-600 hover:text-gray-900"
        >
          Created At
          {{
            asc: " ↑",
            desc: " ↓",
          }[column.getIsSorted() as string] ?? " ↕"}
        </div>
      );
    },
    accessorKey: "model_info.created_at",
    sortingFn: "datetime",
    cell: ({ row }) => {
      const model = row.original;
      return (
        <span className="text-xs">
          {model.model_info.created_at ? new Date(model.model_info.created_at).toLocaleDateString() : "-"}
        </span>
      );
    },
  },
  {
    header: "Created By",
    cell: ({ row }) => {
      const model = row.original;
      return (
        <span className="text-xs">
          {model.model_info.created_by || "-"}
        </span>
      );
    },
  },
  {
    header: ({ column }) => (
      <div
        onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        className="flex items-center gap-2 text-xs cursor-pointer text-gray-600 hover:text-gray-900"
      >
        Input Cost (per 1M tokens)
        {{
          asc: " ↑",
          desc: " ↓",
        }[column.getIsSorted() as string] ?? " ↕"}
      </div>
    ),
    accessorKey: "input_cost",
    cell: ({ row }) => {
      const model = row.original;
      return (
        <pre className="text-xs">
          {model.input_cost || "-"}
        </pre>
      );
    },
  },
  {
    header: ({ column }) => (
      <div
        onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        className="flex items-center gap-2 text-xs cursor-pointer text-gray-600 hover:text-gray-900"
      >
        Output Cost (per 1M tokens)
        {{
          asc: " ↑",
          desc: " ↓",
        }[column.getIsSorted() as string] ?? " ↕"}
      </div>
    ),
    accessorKey: "output_cost",
    cell: ({ row }) => {
      const model = row.original;
      return (
        <pre className="text-xs">
          {model.output_cost || "-"}
        </pre>
      );
    },
  },
  {
    header: "API Base",
    cell: ({ row }) => {
      const model = row.original;
      return (
        <div className="max-w-[200px]">
          <Tooltip title={model.api_base}>
            <pre className="text-xs truncate">
              {model.api_base || "-"}
            </pre>
          </Tooltip>
        </div>
      );
    },
  },
  {
    header: ({ column }) => (
      <div
        onClick={() => column.toggleSorting(column.getIsSorted() === "asc")}
        className="flex items-center gap-2 text-xs cursor-pointer text-gray-600 hover:text-gray-900"
      >
        Team ID
        {{
          asc: " ↑",
          desc: " ↓",
        }[column.getIsSorted() as string] ?? " ↕"}
      </div>
    ),
    accessorKey: "model_info.team_id",
    cell: ({ row }) => {
      const model = row.original;
      return model.model_info.team_id ? (
        <div className="max-w-[200px]">
          <Button
            size="xs"
            variant="light"
            className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 truncate w-full"
            onClick={() => setSelectedTeamId(model.model_info.team_id)}
          >
            {model.model_info.team_id}
          </Button>
        </div>
      ) : (
        "-"
      );
    },
  },
  {
    header: "Status",
    accessorKey: "model_info.db_model",
    cell: ({ row }) => {
      const model = row.original;
      return (
        <div className={`
          inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium
          ${model.model_info.db_model 
            ? 'bg-blue-50 text-blue-600' 
            : 'bg-gray-100 text-gray-600'}
        `}>
          {model.model_info.db_model ? "DB Model" : "Config Model"}
        </div>
      );
    },
  },
];