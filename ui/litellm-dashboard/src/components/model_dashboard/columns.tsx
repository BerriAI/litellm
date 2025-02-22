import { ColumnDef } from "@tanstack/react-table";
import { Button, Badge, Icon } from "@tremor/react";
import { Tooltip } from "antd";
import { getProviderLogoAndName } from "../provider_info_helpers";
import { ModelData } from "./types";
import { TrashIcon, PencilIcon, PencilAltIcon } from "@heroicons/react/outline";
import DeleteModelButton from "../delete_model_button";

export const columns = (
  premiumUser: boolean,
  setSelectedModelId: (id: string) => void,
  setSelectedTeamId: (id: string) => void,
  getDisplayModelName: (model: any) => string,
  handleEditClick: (model: any) => void,
  handleRefreshClick: () => void,
  setEditModel: (edit: boolean) => void,
): ColumnDef<ModelData>[] => [
  {
    header: "Model ID",
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
    header: "Public Model Name",
    accessorKey: "model_name",
    cell: ({ row }) => {
      const displayName = getDisplayModelName(row.original) || "-";
      return (
        <Tooltip title={displayName}>
          <p className="text-xs">
            {displayName.length > 20 ? displayName.slice(0, 20) + "..." : displayName}
          </p>
        </Tooltip>
      );
    },
  },
  {
    header: "Provider",
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
    accessorKey: "litellm_model_name",
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
    header: "Created At",
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
    header: "Updated At",
    accessorKey: "model_info.updated_at",
    sortingFn: "datetime",
    cell: ({ row }) => {
      const model = row.original;
      return (
        <span className="text-xs">
          {model.model_info.updated_at ? new Date(model.model_info.updated_at).toLocaleDateString() : "-"}
        </span>
      );
    },
  },
  {
    header: "Created By",
    accessorKey: "model_info.created_by",
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
    header: () => (
      <Tooltip title="Cost per 1M tokens">
        <span>Input Cost</span>
      </Tooltip>
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
    header: () => (
      <Tooltip title="Cost per 1M tokens">
        <span>Output Cost</span>
      </Tooltip>
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
    header: "Team ID",
    accessorKey: "model_info.team_id",
    cell: ({ row }) => {
      const model = row.original;
      return model.model_info.team_id ? (
        <div className="overflow-hidden">
          <Tooltip title={model.model_info.team_id}>
            <Button
              size="xs"
              variant="light"
              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left overflow-hidden truncate max-w-[200px]"
              onClick={() => setSelectedTeamId(model.model_info.team_id)}
            >
              {model.model_info.team_id.slice(0, 7)}...
            </Button>
          </Tooltip>
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
  {
    id: "actions",
    header: "",
    cell: ({ row }) => {
      const model = row.original;
      return (
        <div className="flex items-center justify-end gap-2 pr-4">
          <Icon
            icon={PencilAltIcon}
            size="sm"
            onClick={() => {
              setSelectedModelId(model.model_info.id);
              setEditModel(true);
            }}
          />
          <Icon
            icon={TrashIcon}
            size="sm"
            onClick={() => {
              setSelectedModelId(model.model_info.id);
              setEditModel(false);
            }}
          />
        </div>
      );
    },
  },
];