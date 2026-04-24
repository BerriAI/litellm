import React from "react";
import {
  Pencil as EditOutlined,
  Info as InfoCircleOutlined,
  RefreshCcw as SyncOutlined,
  Trash2,
} from "lucide-react";
import { ColumnDef } from "@tanstack/react-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import { Separator } from "@/components/ui/separator";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ModelData } from "../../model_dashboard/types";
import { ProviderLogo } from "./ProviderLogo";

// Simple hoverable tooltip wrapper — mirrors antd Tooltip's `title` prop.
const WithTooltip: React.FC<{
  title: React.ReactNode;
  children: React.ReactNode;
  asChild?: boolean;
}> = ({ title, children, asChild = true }) => (
  <TooltipProvider>
    <Tooltip>
      <TooltipTrigger asChild={asChild}>{children}</TooltipTrigger>
      <TooltipContent className="max-w-sm">{title}</TooltipContent>
    </Tooltip>
  </TooltipProvider>
);

const CredentialsInfoPopoverContent: React.FC = () => (
  <div className="flex flex-col gap-3">
    <span className="text-[13px] font-semibold">Credential types</span>
    <div className="flex flex-col gap-2">
      <div className="flex items-start gap-2">
        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-2">
            <SyncOutlined className="h-3.5 w-3.5 text-blue-500" />
            <h5 className="m-0 text-blue-500 text-sm font-semibold">
              Reusable
            </h5>
          </div>
          <span className="text-muted-foreground text-xs">
            Credentials saved in LiteLLM that can be added to models
            repeatedly.
          </span>
        </div>
      </div>
      <Separator />
      <div className="flex items-start gap-2">
        <div className="flex flex-col gap-2">
          <div className="flex items-center gap-2">
            <EditOutlined className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
            <h5 className="m-0 text-sm font-semibold">Manual</h5>
          </div>
          <span className="text-muted-foreground text-xs">
            Credentials added directly during model creation or defined in
            the config file.
          </span>
        </div>
      </div>
    </div>
  </div>
);

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const columns = (
  userRole: string,
  userID: string,
  premiumUser: boolean,
  setSelectedModelId: (id: string) => void,
  setSelectedTeamId: (id: string) => void,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  getDisplayModelName: (model: any) => string,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  handleEditClick: (model: any) => void,
  handleRefreshClick: () => void,
  expandedRows: Set<string>,
  setExpandedRows: (expandedRows: Set<string>) => void,
  onDeleteClick?: (modelId: string) => void,
): ColumnDef<ModelData>[] => [
  {
    header: () => <span className="text-sm font-semibold">Model ID</span>,
    accessorKey: "model_info.id",
    enableSorting: false,
    size: 130,
    minSize: 80,
    cell: ({ row }) => {
      const model = row.original;
      return (
        <WithTooltip title={model.model_info.id}>
          <span
            className="truncate text-blue-500 bg-blue-50 hover:bg-blue-100 cursor-pointer w-full block text-sm px-2"
            onClick={(e) => {
              e.stopPropagation();
              setSelectedModelId(model.model_info.id);
            }}
          >
            {model.model_info.id}
          </span>
        </WithTooltip>
      );
    },
  },
  {
    header: () => (
      <span className="text-sm font-semibold">Model Information</span>
    ),
    accessorKey: "model_name",
    size: 250,
    minSize: 120,
    cell: ({ row }) => {
      const model = row.original;
      const displayName = getDisplayModelName(row.original) || "-";
      const popoverContent = (
        <div className="flex flex-col gap-3 min-w-[220px]">
          <div className="flex items-center gap-2">
            <ProviderLogo provider={model.provider} />
            <span className="text-muted-foreground text-xs truncate">
              {model.provider || "Unknown provider"}
            </span>
          </div>

          <div className="flex flex-col gap-1.5">
            <div className="flex flex-col gap-0.5 w-full">
              <span className="text-muted-foreground text-[11px]">
                Public Model Name
              </span>
              <span
                className="font-semibold text-[13px] max-w-[480px] truncate"
                title={displayName}
              >
                {displayName}
              </span>
            </div>

            <div className="flex flex-col gap-0.5">
              <span className="text-muted-foreground text-[11px]">
                LiteLLM Model Name
              </span>
              <span
                className="text-[13px] truncate"
                title={model.litellm_model_name || "-"}
              >
                {model.litellm_model_name || "-"}
              </span>
            </div>
          </div>
        </div>
      );

      return (
        <Popover>
          <PopoverTrigger asChild>
            <div className="flex items-start space-x-2 min-w-0 w-full cursor-pointer">
              <div className="flex-shrink-0 mt-0.5">
                {model.provider ? (
                  <ProviderLogo provider={model.provider} />
                ) : (
                  <div className="w-4 h-4 rounded-full bg-muted flex items-center justify-center text-xs">
                    -
                  </div>
                )}
              </div>

              <div className="flex flex-col min-w-0 flex-1">
                <span className="truncate text-foreground text-xs font-medium leading-4">
                  {displayName}
                </span>
                <span className="truncate text-muted-foreground text-xs leading-4 mt-[2px]">
                  {model.litellm_model_name || "-"}
                </span>
              </div>
            </div>
          </PopoverTrigger>
          <PopoverContent side="right" className="max-w-[500px]">
            {popoverContent}
          </PopoverContent>
        </Popover>
      );
    },
  },
  {
    header: () => (
      <span className="flex items-center gap-1">
        <span className="text-sm font-semibold">Credentials</span>
        <Popover>
          <PopoverTrigger asChild>
            <button
              type="button"
              className="cursor-pointer text-muted-foreground hover:text-foreground"
            >
              <InfoCircleOutlined className="h-3 w-3" />
            </button>
          </PopoverTrigger>
          <PopoverContent side="bottom">
            <CredentialsInfoPopoverContent />
          </PopoverContent>
        </Popover>
      </span>
    ),
    accessorKey: "litellm_credential_name",
    enableSorting: false,
    size: 180,
    minSize: 100,
    cell: ({ row }) => {
      const model = row.original;
      const credentialName = model.litellm_params?.litellm_credential_name;
      const isReusable = !!credentialName;

      return (
        <div className="flex items-center space-x-2 min-w-0 w-full">
          {isReusable ? (
            <>
              <SyncOutlined className="flex-shrink-0 h-3.5 w-3.5 text-blue-500" />
              <span
                className="text-xs truncate text-blue-600"
                title={credentialName}
              >
                {credentialName}
              </span>
            </>
          ) : (
            <>
              <EditOutlined className="flex-shrink-0 h-3.5 w-3.5 text-muted-foreground" />
              <span className="text-xs text-muted-foreground">Manual</span>
            </>
          )}
        </div>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Created By</span>,
    accessorKey: "model_info.created_by",
    sortingFn: "datetime",
    size: 160,
    minSize: 100,
    cell: ({ row }) => {
      const model = row.original;
      const isConfigModel = !model.model_info?.db_model;
      const createdBy = model.model_info.created_by;
      const createdAt = model.model_info.created_at
        ? new Date(model.model_info.created_at).toLocaleDateString()
        : null;

      return (
        <div className="flex flex-col min-w-0 w-full">
          <div
            className="text-xs font-medium text-foreground truncate"
            title={isConfigModel ? "Defined in config" : createdBy || "Unknown"}
          >
            {isConfigModel ? "Defined in config" : createdBy || "Unknown"}
          </div>
          <div
            className="text-xs text-muted-foreground truncate mt-0.5"
            title={
              isConfigModel ? "Config file" : createdAt || "Unknown date"
            }
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
    size: 120,
    minSize: 80,
    cell: ({ row }) => {
      const model = row.original;
      return (
        <span className="text-xs">
          {model.model_info.updated_at
            ? new Date(model.model_info.updated_at).toLocaleDateString()
            : "-"}
        </span>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Costs</span>,
    accessorKey: "input_cost",
    size: 120,
    minSize: 80,
    cell: ({ row }) => {
      const model = row.original;
      const inputCost = model.input_cost;
      const outputCost = model.output_cost;

      if (inputCost == null && outputCost == null) {
        return (
          <div className="w-full">
            <span className="text-xs text-muted-foreground">-</span>
          </div>
        );
      }

      return (
        <WithTooltip title="Cost per 1M tokens">
          <div className="flex flex-col min-w-0 w-full">
            {inputCost != null && (
              <div className="text-xs font-medium text-foreground truncate">
                In: ${inputCost}
              </div>
            )}
            {outputCost != null && (
              <div className="text-xs text-muted-foreground truncate mt-0.5">
                Out: ${outputCost}
              </div>
            )}
          </div>
        </WithTooltip>
      );
    },
  },
  {
    header: () => <span className="text-sm font-semibold">Team ID</span>,
    accessorKey: "model_info.team_id",
    enableSorting: false,
    size: 130,
    minSize: 80,
    cell: ({ row }) => {
      const model = row.original;
      return model.model_info.team_id ? (
        <div className="overflow-hidden w-full">
          <WithTooltip title={model.model_info.team_id}>
            <Button
              variant="ghost"
              size="sm"
              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 h-auto text-left overflow-hidden truncate w-full justify-start"
              onClick={(e) => {
                e.stopPropagation();
                setSelectedTeamId(model.model_info.team_id);
              }}
            >
              {model.model_info.team_id.slice(0, 7)}...
            </Button>
          </WithTooltip>
        </div>
      ) : (
        "-"
      );
    },
  },
  {
    header: () => (
      <span className="text-sm font-semibold">Model Access Group</span>
    ),
    accessorKey: "model_info.model_access_group",
    enableSorting: false,
    size: 180,
    minSize: 100,
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
        <div className="flex items-center gap-1 overflow-hidden w-full">
          <Badge
            variant="secondary"
            className="bg-blue-100 text-blue-700 text-xs px-1.5 py-0.5 h-5 leading-tight flex-shrink-0"
          >
            {accessGroups[0]}
          </Badge>

          {(isExpanded ||
            (!shouldShowExpandButton && accessGroups.length === 2)) &&
            accessGroups.slice(1).map((group: string, index: number) => (
              <Badge
                key={index + 1}
                variant="secondary"
                className="bg-blue-100 text-blue-700 text-xs px-1.5 py-0.5 h-5 leading-tight flex-shrink-0"
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
    size: 120,
    minSize: 80,
    cell: ({ row }) => {
      const model = row.original;
      return (
        <div
          className={`
          inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium
          ${model.model_info.db_model ? "bg-blue-50 text-blue-600" : "bg-muted text-muted-foreground"}
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
    size: 60,
    minSize: 40,
    enableResizing: false,
    cell: ({ row }) => {
      const model = row.original;
      const canEditModel =
        userRole === "Admin" || model.model_info?.created_by === userID;
      const isConfigModel = !model.model_info?.db_model;
      return (
        <div className="flex items-center justify-end gap-2 pr-4">
          {isConfigModel ? (
            <WithTooltip
              title="Config model cannot be deleted on the dashboard. Please delete it from the config file."
            >
              <span className="inline-flex items-center opacity-50 cursor-not-allowed">
                <Trash2 className="h-4 w-4" />
              </span>
            </WithTooltip>
          ) : (
            <WithTooltip title="Delete model">
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  if (canEditModel && onDeleteClick) {
                    onDeleteClick(model.model_info.id);
                  }
                }}
                className={
                  !canEditModel
                    ? "opacity-50 cursor-not-allowed"
                    : "cursor-pointer hover:text-destructive"
                }
                aria-label="Delete model"
              >
                <Trash2 className="h-4 w-4" />
              </button>
            </WithTooltip>
          )}
        </div>
      );
    },
  },
];
