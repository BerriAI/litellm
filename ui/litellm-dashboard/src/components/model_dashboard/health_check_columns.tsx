import { ColumnDef } from "@tanstack/react-table";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";
import { Info, Play, RefreshCcw } from "lucide-react";
import { Team } from "@/components/key_team_helpers/key_list";

interface HealthCheckData {
  model_name: string;
  model_info: {
    id: string;
    created_at?: string;
    team_id?: string;
  };
  provider?: string;
  litellm_model_name?: string;
  health_status: string;
  last_check: string;
  last_success: string;
  health_loading: boolean;
  health_error?: string;
  health_full_error?: string;
}

interface HealthStatus {
  status: string;
  lastCheck: string;
  lastSuccess?: string;
  loading: boolean;
  error?: string;
  fullError?: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  successResponse?: any;
}

export const healthCheckColumns = (
  modelHealthStatuses: { [key: string]: HealthStatus },
  selectedModelsForHealth: string[],
  allModelsSelected: boolean,
  handleModelSelection: (modelId: string, checked: boolean) => void,
  handleSelectAll: (checked: boolean) => void,
  runIndividualHealthCheck: (modelId: string) => void,
  // eslint-disable-next-line no-undef
  getStatusBadge: (status: string) => JSX.Element,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  getDisplayModelName: (model: any) => string,
  showErrorModal?: (
    modelName: string,
    cleanedError: string,
    fullError: string,
  ) => void,
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  showSuccessModal?: (modelName: string, response: any) => void,
  setSelectedModelId?: (modelId: string) => void,
  teams?: Team[] | null,
): ColumnDef<HealthCheckData>[] => [
  {
    header: () => (
      <div className="flex items-center gap-2">
        <Checkbox
          checked={
            selectedModelsForHealth.length > 0 && !allModelsSelected
              ? "indeterminate"
              : allModelsSelected
                ? true
                : false
          }
          onCheckedChange={(c) => handleSelectAll(c === true)}
          onClick={(e) => e.stopPropagation()}
        />
        <span>Model ID</span>
      </div>
    ),
    accessorKey: "model_info.id",
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => {
      const model = row.original;
      const modelId = model.model_info?.id ?? "";
      const isSelected = selectedModelsForHealth.includes(modelId);

      return (
        <div className="flex items-center gap-2">
          <Checkbox
            checked={isSelected}
            onCheckedChange={(c) => handleModelSelection(modelId, c === true)}
            onClick={(e) => e.stopPropagation()}
          />
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  type="button"
                  className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 dark:bg-blue-950/30 dark:hover:bg-blue-950/60 text-xs font-normal px-2 py-0.5 text-left w-full truncate whitespace-nowrap cursor-pointer max-w-[15ch] rounded"
                  onClick={() =>
                    setSelectedModelId && setSelectedModelId(modelId)
                  }
                >
                  {modelId}
                </button>
              </TooltipTrigger>
              <TooltipContent>{modelId}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      );
    },
  },
  {
    header: "Model Name",
    accessorKey: "model_name",
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => {
      const model = row.original;
      const displayName = getDisplayModelName(model) || model.model_name;

      return (
        <div className="font-medium text-sm">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="truncate max-w-[200px]">{displayName}</div>
              </TooltipTrigger>
              <TooltipContent>{displayName}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      );
    },
  },
  {
    header: "Team Alias",
    accessorKey: "model_info.team_id",
    enableSorting: true,
    sortingFn: "alphanumeric",
    cell: ({ row }) => {
      const model = row.original;
      const teamId = model.model_info?.team_id;

      if (!teamId) {
        return <span className="text-muted-foreground text-sm">-</span>;
      }

      const team = teams?.find((t) => t.team_id === teamId);
      const teamAlias = team?.team_alias || teamId;

      return (
        <div className="text-sm">
          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="truncate max-w-[150px]">{teamAlias}</div>
              </TooltipTrigger>
              <TooltipContent>{teamAlias}</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      );
    },
  },
  {
    header: "Health Status",
    accessorKey: "health_status",
    enableSorting: true,
    sortingFn: (rowA, rowB) => {
      const statusA = (rowA.getValue("health_status") as string) || "unknown";
      const statusB = (rowB.getValue("health_status") as string) || "unknown";

      const statusOrder = { healthy: 0, checking: 1, unknown: 2, unhealthy: 3 };
      const orderA = statusOrder[statusA as keyof typeof statusOrder] ?? 4;
      const orderB = statusOrder[statusB as keyof typeof statusOrder] ?? 4;

      return orderA - orderB;
    },
    cell: ({ row }) => {
      const model = row.original;
      const healthStatus = {
        status: model.health_status,
        loading: model.health_loading,
        error: model.health_error,
      };

      if (healthStatus.loading) {
        return (
          <div className="flex items-center space-x-2">
            <div className="flex space-x-1">
              <div className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse"></div>
              <div
                className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse"
                style={{ animationDelay: "0.2s" }}
              ></div>
              <div
                className="w-2 h-2 bg-indigo-500 rounded-full animate-pulse"
                style={{ animationDelay: "0.4s" }}
              ></div>
            </div>
            <span className="text-muted-foreground text-sm">Checking...</span>
          </div>
        );
      }

      const modelId = model.model_info?.id ?? "";
      const displayName = getDisplayModelName(model) || model.model_name;
      const hasSuccessResponse =
        healthStatus.status === "healthy" &&
        modelHealthStatuses[modelId]?.successResponse;

      return (
        <div className="flex items-center space-x-2">
          {getStatusBadge(healthStatus.status)}
          {hasSuccessResponse && showSuccessModal && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() =>
                      showSuccessModal(
                        displayName,
                        modelHealthStatuses[modelId]?.successResponse,
                      )
                    }
                    className="p-1 text-emerald-600 hover:text-emerald-800 hover:bg-emerald-50 dark:hover:bg-emerald-950/30 rounded cursor-pointer transition-colors"
                    aria-label="View response details"
                  >
                    <Info className="h-4 w-4" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>View response details</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
      );
    },
  },
  {
    header: "Error Details",
    accessorKey: "health_error",
    enableSorting: false,
    cell: ({ row }) => {
      const model = row.original;
      const modelId = model.model_info?.id ?? "";
      const displayName = getDisplayModelName(model) || model.model_name;
      const healthStatus = modelHealthStatuses[modelId];

      if (!healthStatus?.error) {
        return <span className="text-muted-foreground text-sm">No errors</span>;
      }

      const cleanedError = healthStatus.error;
      const fullError = healthStatus.fullError || healthStatus.error;

      return (
        <div className="flex items-center space-x-2">
          <div className="max-w-[200px]">
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <span className="text-red-600 text-sm truncate block">
                    {cleanedError}
                  </span>
                </TooltipTrigger>
                <TooltipContent>{cleanedError}</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          </div>
          {showErrorModal && fullError !== cleanedError && (
            <TooltipProvider>
              <Tooltip>
                <TooltipTrigger asChild>
                  <button
                    type="button"
                    onClick={() =>
                      showErrorModal(displayName, cleanedError, fullError)
                    }
                    className="p-1 text-red-600 hover:text-red-800 hover:bg-red-50 dark:hover:bg-red-950/30 rounded cursor-pointer transition-colors"
                    aria-label="View full error details"
                  >
                    <Info className="h-4 w-4" />
                  </button>
                </TooltipTrigger>
                <TooltipContent>View full error details</TooltipContent>
              </Tooltip>
            </TooltipProvider>
          )}
        </div>
      );
    },
  },
  {
    header: "Last Check",
    accessorKey: "last_check",
    enableSorting: true,
    sortingFn: (rowA, rowB) => {
      const lastCheckA =
        (rowA.getValue("last_check") as string) || "Never checked";
      const lastCheckB =
        (rowB.getValue("last_check") as string) || "Never checked";

      if (lastCheckA === "Never checked" && lastCheckB === "Never checked")
        return 0;
      if (lastCheckA === "Never checked") return 1;
      if (lastCheckB === "Never checked") return -1;
      if (
        lastCheckA === "Check in progress..." &&
        lastCheckB === "Check in progress..."
      )
        return 0;
      if (lastCheckA === "Check in progress...") return -1;
      if (lastCheckB === "Check in progress...") return 1;

      const dateA = new Date(lastCheckA);
      const dateB = new Date(lastCheckB);

      if (isNaN(dateA.getTime()) && isNaN(dateB.getTime())) return 0;
      if (isNaN(dateA.getTime())) return 1;
      if (isNaN(dateB.getTime())) return -1;

      return dateB.getTime() - dateA.getTime();
    },
    cell: ({ row }) => {
      const model = row.original;

      return (
        <span className="text-muted-foreground text-sm">
          {model.health_loading ? "Check in progress..." : model.last_check}
        </span>
      );
    },
  },
  {
    header: "Last Success",
    accessorKey: "last_success",
    enableSorting: true,
    sortingFn: (rowA, rowB) => {
      const lastSuccessA =
        (rowA.getValue("last_success") as string) || "Never succeeded";
      const lastSuccessB =
        (rowB.getValue("last_success") as string) || "Never succeeded";

      if (lastSuccessA === "Never succeeded" && lastSuccessB === "Never succeeded")
        return 0;
      if (lastSuccessA === "Never succeeded") return 1;
      if (lastSuccessB === "Never succeeded") return -1;
      if (lastSuccessA === "None" && lastSuccessB === "None") return 0;
      if (lastSuccessA === "None") return 1;
      if (lastSuccessB === "None") return -1;

      const dateA = new Date(lastSuccessA);
      const dateB = new Date(lastSuccessB);

      if (isNaN(dateA.getTime()) && isNaN(dateB.getTime())) return 0;
      if (isNaN(dateA.getTime())) return 1;
      if (isNaN(dateB.getTime())) return -1;

      return dateB.getTime() - dateA.getTime();
    },
    cell: ({ row }) => {
      const model = row.original;
      const modelId = model.model_info?.id ?? "";
      const healthStatus = modelHealthStatuses[modelId];
      const lastSuccess = healthStatus?.lastSuccess || "None";

      return (
        <span className="text-muted-foreground text-sm">{lastSuccess}</span>
      );
    },
  },
  {
    header: "Actions",
    id: "actions",
    cell: ({ row }) => {
      const model = row.original;
      const modelId = model.model_info?.id ?? "";

      const hasExistingStatus =
        model.health_status && model.health_status !== "none";
      const tooltipText = model.health_loading
        ? "Checking..."
        : hasExistingStatus
          ? "Re-run Health Check"
          : "Run Health Check";

      return (
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                data-testid="run-health-check-btn"
                className={cn(
                  "p-2 rounded-md transition-colors",
                  model.health_loading
                    ? "text-muted-foreground cursor-not-allowed bg-muted"
                    : "text-indigo-600 hover:text-indigo-700 hover:bg-indigo-50 dark:hover:bg-indigo-950/30",
                )}
                onClick={() => {
                  if (!model.health_loading) {
                    runIndividualHealthCheck(modelId);
                  }
                }}
                disabled={model.health_loading}
              >
                {model.health_loading ? (
                  <div className="flex space-x-1">
                    <div className="w-1 h-1 bg-muted-foreground rounded-full animate-pulse"></div>
                    <div
                      className="w-1 h-1 bg-muted-foreground rounded-full animate-pulse"
                      style={{ animationDelay: "0.2s" }}
                    ></div>
                    <div
                      className="w-1 h-1 bg-muted-foreground rounded-full animate-pulse"
                      style={{ animationDelay: "0.4s" }}
                    ></div>
                  </div>
                ) : hasExistingStatus ? (
                  <RefreshCcw className="h-4 w-4" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
              </button>
            </TooltipTrigger>
            <TooltipContent>{tooltipText}</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      );
    },
    enableSorting: false,
  },
];
