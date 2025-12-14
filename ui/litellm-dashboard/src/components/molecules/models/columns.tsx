import { KeyIcon, TrashIcon } from "@heroicons/react/outline";
import { ColumnDef } from "@tanstack/react-table";
import { Badge, Button, Icon } from "@tremor/react";
import { Tooltip } from "antd";
import { ModelData } from "../../model_dashboard/types";
import { ProviderLogo } from "./ProviderLogo";

export const columns = (
  userRole: string,
  userID: string,
  premiumUser: boolean,
  setSelectedModelId: (id: string) => void,
  setSelectedTeamId: (id: string) => void,
  getDisplayModelName: (model: any) => string,
  handleEditClick: (model: any) => void,
  handleRefreshClick: () => void,
  setEditModel: (edit: boolean) => void,
  expandedRows: Set<string>,
  setExpandedRows: (expandedRows: Set<string>) => void,
): ColumnDef<ModelData>[] => [
  {
    header: () => <span className="text-sm font-semibold">Model ID</span>,
    accessorKey: "model_info.id",
    cell: ({ row }) => {
      const model = row.original;
      return (
        <Tooltip title={model.model_info.id}>
          <div
            className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left w-full truncate whitespace-nowrap cursor-pointer max-w-[15ch]"
            onClick={() => setSelectedModelId(model.model_info.id)}
          >
            {model.model_info.id}
          </div>
        </Tooltip>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Model Information</span>,
    accessorKey: "model_name",
    size: 250, // Fixed column width
    cell: ({ row }) => {
      const model = row.original;
      const displayName = getDisplayModelName(row.original) || "-";
      const tooltipContent = (
        <div>
          <div>
            <strong>Provider:</strong> {model.provider || "-"}
          </div>
          <div>
            <strong>Public Model Name:</strong> {displayName}
          </div>
          <div>
            <strong>LiteLLM Model Name:</strong> {model.litellm_model_name || "-"}
          </div>
        </div>
      );

      return (
        <Tooltip title={tooltipContent}>
          <div className="flex items-start space-x-2 min-w-0 w-full max-w-[250px]">
            {/* Provider Icon */}
            <div className="flex-shrink-0 mt-0.5">
              {model.provider ? (
                <ProviderLogo provider={model.provider} />
              ) : (
                <div className="w-4 h-4 rounded-full bg-gray-200 flex items-center justify-center text-xs">-</div>
              )}
            </div>

            {/* Model Names Container */}
            <div className="flex flex-col min-w-0 flex-1">
              {/* Public Model Name */}
              <div className="text-xs font-medium text-gray-900 truncate max-w-[210px]">{displayName}</div>
              {/* LiteLLM Model Name */}
              <div className="text-xs text-gray-500 truncate mt-0.5 max-w-[210px]">
                {model.litellm_model_name || "-"}
              </div>
            </div>
          </div>
        </Tooltip>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Credentials</span>,
    accessorKey: "litellm_credential_name",
    size: 180, // Fixed column width
    cell: ({ row }) => {
      const model = row.original;
      const credentialName = model.litellm_params?.litellm_credential_name;

      return credentialName ? (
        <Tooltip title={`Credential: ${credentialName}`}>
          <div className="flex items-center space-x-2 max-w-[180px]">
            <KeyIcon className="w-4 h-4 text-blue-500 flex-shrink-0" />
            <span className="text-xs truncate" title={credentialName}>
              {credentialName}
            </span>
          </div>
        </Tooltip>
      ) : (
        <div className="flex items-center space-x-2 max-w-[180px]">
          <KeyIcon className="w-4 h-4 text-gray-300 flex-shrink-0" />
          <span className="text-xs text-gray-400">No credentials</span>
        </div>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Created By</span>,
    accessorKey: "model_info.created_by",
    sortingFn: "datetime",
    size: 160, // Fixed column width
    cell: ({ row }) => {
      const model = row.original;
      const isConfigModel = !model.model_info?.db_model;
      const createdBy = model.model_info.created_by;
      const createdAt = model.model_info.created_at ? new Date(model.model_info.created_at).toLocaleDateString() : null;

      return (
        <div className="flex flex-col min-w-0 max-w-[160px]">
          {/* Created By - Primary */}
          <div
            className="text-xs font-medium text-gray-900 truncate"
            title={isConfigModel ? "Defined in config" : createdBy || "Unknown"}
          >
            {isConfigModel ? "Defined in config" : createdBy || "Unknown"}
          </div>
          {/* Created At - Secondary */}
          <div
            className="text-xs text-gray-500 truncate mt-0.5"
            title={isConfigModel ? "Config file" : createdAt || "Unknown date"}
          >
            {isConfigModel ? "-" : createdAt || "Unknown date"}
          </div>
        </div>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Updated At</span>,
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
    header: () => <span className="text-sm font-semibold">Costs</span>,
    accessorKey: "input_cost",
    size: 120, // Fixed column width
    cell: ({ row }) => {
      const model = row.original;
      const inputCost = model.input_cost;
      const outputCost = model.output_cost;

      // If both costs are missing or undefined, show "-"
      if (!inputCost && !outputCost) {
        return (
          <div className="max-w-[120px]">
            <span className="text-xs text-gray-400">-</span>
          </div>
        );
      }

      return (
        <Tooltip title="Cost per 1M tokens">
          <div className="flex flex-col min-w-0 max-w-[120px]">
            {/* Input Cost - Primary */}
            {inputCost && <div className="text-xs font-medium text-gray-900 truncate">In: ${inputCost}</div>}
            {/* Output Cost - Secondary */}
            {outputCost && <div className="text-xs text-gray-500 truncate mt-0.5">Out: ${outputCost}</div>}
          </div>
        </Tooltip>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Team ID</span>,
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
    header: () => <span className="text-sm font-semibold">Model Access Group</span>,
    accessorKey: "model_info.model_access_group",
    enableSorting: false,
    cell: ({ row }) => {
      const model = row.original;
      const accessGroups = model.model_info.access_groups;

      if (!accessGroups || accessGroups.length === 0) {
        return "-";
      }

      const modelId = model.model_info.id;
      const isExpanded = expandedRows.has(modelId);
      const shouldShowExpandButton = accessGroups.length > 1;

      const toggleExpanded = () => {
        const newExpanded = new Set(expandedRows);
        if (isExpanded) {
          newExpanded.delete(modelId);
        } else {
          newExpanded.add(modelId);
        }
        setExpandedRows(newExpanded);
      };

      return (
        <div className="flex items-center gap-1 overflow-hidden">
          <Badge size="xs" color="blue" className="text-xs px-1.5 py-0.5 h-5 leading-tight flex-shrink-0">
            {accessGroups[0]}
          </Badge>

          {(isExpanded || (!shouldShowExpandButton && accessGroups.length === 2)) &&
            accessGroups.slice(1).map((group: string, index: number) => (
              <Badge
                key={index + 1}
                size="xs"
                color="blue"
                className="text-xs px-1.5 py-0.5 h-5 leading-tight flex-shrink-0"
              >
                {group}
              </Badge>
            ))}

          {shouldShowExpandButton && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                toggleExpanded();
              }}
              className="text-xs text-blue-600 hover:text-blue-800 px-1 py-0.5 rounded hover:bg-blue-50 h-5 leading-tight flex-shrink-0 whitespace-nowrap"
            >
              {isExpanded ? "−" : `+${accessGroups.length - 1}`}
            </button>
          )}
        </div>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Status</span>,
    accessorKey: "model_info.db_model",
    cell: ({ row }) => {
      const model = row.original;
      return (
        <div
          className={`
          inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium
          ${model.model_info.db_model ? "bg-blue-50 text-blue-600" : "bg-gray-100 text-gray-600"}
        `}
        >
          {model.model_info.db_model ? "DB Model" : "Config Model"}
        </div>
      );
    },
  },
  {
    id: "actions",
    header: () => <span className="text-sm font-semibold">Actions</span>,
    cell: ({ row }) => {
      const model = row.original;
      const canEditModel = userRole === "Admin" || model.model_info?.created_by === userID;
      const isConfigModel = !model.model_info?.db_model;
      return (
        <div className="flex items-center justify-end gap-2 pr-4">
          {isConfigModel ? (
            <Tooltip title="Config model cannot be deleted on the dashboard. Please delete it from the config file.">
              <Icon icon={TrashIcon} size="sm" className="opacity-50 cursor-not-allowed" />
            </Tooltip>
          ) : (
            <Tooltip title="Delete model">
              <Icon
                icon={TrashIcon}
                size="sm"
                onClick={() => {
                  if (canEditModel) {
                    setSelectedModelId(model.model_info.id);
                    setEditModel(false);
                  }
                }}
                className={!canEditModel ? "opacity-50 cursor-not-allowed" : "cursor-pointer hover:text-red-600"}
              />
            </Tooltip>
          )}
        </div>
      );
    },
  },
];
