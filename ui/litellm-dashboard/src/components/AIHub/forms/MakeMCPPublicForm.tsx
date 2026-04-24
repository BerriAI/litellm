import React, { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { cn } from "@/lib/utils";
import { Check } from "lucide-react";
import { makeMCPPublicCall } from "../../networking";
import NotificationsManager from "../../molecules/notifications_manager";
import { MCPServerData } from "@/components/mcp_hub_table_columns";

interface MakeMCPPublicFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string;
  mcpHubData: MCPServerData[];
  onSuccess: () => void;
}

const STATUS_BADGE_CLASSES = (status?: string): string => {
  if (status === "active" || status === "healthy") {
    return "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300";
  }
  if (status === "inactive" || status === "unhealthy") {
    return "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300";
  }
  return "bg-muted text-muted-foreground";
};

function Stepper({ current, steps }: { current: number; steps: string[] }) {
  return (
    <ol className="flex items-center gap-2 mb-6">
      {steps.map((label, i) => {
        const active = i === current;
        const completed = i < current;
        return (
          <li
            key={label}
            className="flex items-center gap-2 flex-1 min-w-0"
          >
            <div
              className={cn(
                "flex h-6 w-6 items-center justify-center rounded-full border text-xs font-medium",
                completed
                  ? "bg-primary text-primary-foreground border-primary"
                  : active
                    ? "border-primary text-primary"
                    : "border-border text-muted-foreground",
              )}
            >
              {completed ? <Check className="h-3 w-3" /> : i + 1}
            </div>
            <span
              className={cn(
                "text-sm truncate",
                active || completed
                  ? "text-foreground"
                  : "text-muted-foreground",
              )}
            >
              {label}
            </span>
            {i < steps.length - 1 && (
              <div className="h-px flex-1 bg-border" />
            )}
          </li>
        );
      })}
    </ol>
  );
}

const MakeMCPPublicForm: React.FC<MakeMCPPublicFormProps> = ({
  visible,
  onClose,
  accessToken,
  mcpHubData,
  onSuccess,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedServers, setSelectedServers] = useState<Set<string>>(
    new Set(),
  );
  const [loading, setLoading] = useState(false);

  const handleClose = () => {
    setCurrentStep(0);
    setSelectedServers(new Set());
    onClose();
  };

  const handleNext = () => {
    if (currentStep === 0) {
      if (selectedServers.size === 0) {
        NotificationsManager.fromBackend(
          "Please select at least one MCP server to make public",
        );
        return;
      }
      setCurrentStep(1);
    }
  };

  const handlePrevious = () => {
    if (currentStep === 1) {
      setCurrentStep(0);
    }
  };

  const handleServerSelection = (serverId: string, checked: boolean) => {
    const newSelection = new Set(selectedServers);
    if (checked) {
      newSelection.add(serverId);
    } else {
      newSelection.delete(serverId);
    }
    setSelectedServers(newSelection);
  };

  const handleSelectAll = (checked: boolean) => {
    if (checked) {
      const allServerIds = mcpHubData.map((server) => server.server_id);
      setSelectedServers(new Set(allServerIds));
    } else {
      setSelectedServers(new Set());
    }
  };

  useEffect(() => {
    if (visible && mcpHubData.length > 0) {
      const publicServerIds = mcpHubData
        .filter((server) => server.mcp_info?.is_public === true)
        .map((server) => server.server_id);

      setSelectedServers(new Set(publicServerIds));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [visible]);

  const handleSubmit = async () => {
    if (selectedServers.size === 0) {
      NotificationsManager.fromBackend(
        "Please select at least one MCP server to make public",
      );
      return;
    }

    setLoading(true);
    try {
      const serverIdsToMakePublic = Array.from(selectedServers);

      await makeMCPPublicCall(accessToken, serverIdsToMakePublic);

      NotificationsManager.success(
        `Successfully made ${serverIdsToMakePublic.length} MCP server(s) public!`,
      );
      handleClose();
      onSuccess();
    } catch (error) {
      console.error("Error making MCP servers public:", error);
      NotificationsManager.fromBackend(
        "Failed to make MCP servers public. Please try again.",
      );
    } finally {
      setLoading(false);
    }
  };

  const renderStep1Content = () => {
    const allServersSelected =
      mcpHubData.length > 0 &&
      mcpHubData.every((server) => selectedServers.has(server.server_id));
    const isIndeterminate = selectedServers.size > 0 && !allServersSelected;

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">
            Select MCP Servers to Make Public
          </h3>
          <label className="flex items-center gap-2 cursor-pointer">
            <Checkbox
              checked={
                isIndeterminate
                  ? "indeterminate"
                  : allServersSelected
                    ? true
                    : false
              }
              onCheckedChange={(c) => handleSelectAll(c === true)}
              disabled={mcpHubData.length === 0}
            />
            <span className="text-sm">
              Select All {mcpHubData.length > 0 && `(${mcpHubData.length})`}
            </span>
          </label>
        </div>

        <p className="text-sm text-muted-foreground">
          Select the MCP servers you want to be visible on the public model
          hub. Users will still require a valid Virtual Key to use these
          servers.
        </p>

        <div className="max-h-96 overflow-y-auto border border-border rounded-lg p-4">
          <div className="space-y-3">
            {mcpHubData.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <p>No MCP servers available.</p>
              </div>
            ) : (
              mcpHubData.map((server) => {
                const isPublic = server.mcp_info?.is_public === true;
                return (
                  <label
                    key={server.server_id}
                    className={cn(
                      "flex items-center space-x-3 p-3 border border-border rounded-lg hover:bg-muted cursor-pointer",
                    )}
                  >
                    <Checkbox
                      checked={selectedServers.has(server.server_id)}
                      onCheckedChange={(c) =>
                        handleServerSelection(server.server_id, c === true)
                      }
                    />
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 flex-wrap gap-1">
                        <span className="font-medium">
                          {server.server_name}
                        </span>
                        {isPublic && (
                          <Badge className="text-xs bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300">
                            Public
                          </Badge>
                        )}
                        <Badge className="text-xs bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                          {server.transport}
                        </Badge>
                        <Badge
                          className={cn(
                            "text-xs",
                            STATUS_BADGE_CLASSES(server.status),
                          )}
                        >
                          {server.status || "unknown"}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1">
                        {server.description || server.url}
                      </p>
                      {server.allowed_tools &&
                        server.allowed_tools.length > 0 && (
                          <div className="flex flex-wrap gap-1 mt-1">
                            {server.allowed_tools
                              .slice(0, 3)
                              .map((tool, idx) => (
                                <Badge
                                  key={idx}
                                  className="text-[10px] bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300"
                                >
                                  {tool}
                                </Badge>
                              ))}
                            {server.allowed_tools.length > 3 && (
                              <span className="text-xs text-muted-foreground">
                                +{server.allowed_tools.length - 3} more
                              </span>
                            )}
                          </div>
                        )}
                    </div>
                  </label>
                );
              })
            )}
          </div>
        </div>

        {selectedServers.size > 0 && (
          <div className="bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 rounded-lg p-3">
            <p className="text-sm text-blue-800 dark:text-blue-200">
              <strong>{selectedServers.size}</strong> MCP server
              {selectedServers.size !== 1 ? "s" : ""} selected
            </p>
          </div>
        )}
      </div>
    );
  };

  const renderStep2Content = () => {
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">
          Confirm Making MCP Servers Public
        </h3>

        <div className="bg-amber-50 dark:bg-amber-950/30 border border-amber-200 dark:border-amber-900 rounded-lg p-4">
          <p className="text-sm text-amber-800 dark:text-amber-200">
            <strong>Warning:</strong> Once you make these MCP servers public,
            anyone who can go to the <code>/ui/model_hub_table</code> will be
            able to know they exist on the proxy.
          </p>
        </div>

        <div className="space-y-3">
          <p className="font-medium">MCP Servers to be made public:</p>
          <div className="max-h-48 overflow-y-auto border border-border rounded-lg p-3">
            <div className="space-y-2">
              {Array.from(selectedServers).map((serverId) => {
                const server = mcpHubData.find(
                  (s) => s.server_id === serverId,
                );
                return (
                  <div
                    key={serverId}
                    className="flex items-center justify-between p-2 bg-muted rounded"
                  >
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 flex-wrap gap-1">
                        <span className="font-medium">
                          {server?.server_name || serverId}
                        </span>
                        {server && (
                          <>
                            <Badge className="text-[10px] bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                              {server.transport}
                            </Badge>
                            <Badge
                              className={cn(
                                "text-[10px]",
                                STATUS_BADGE_CLASSES(server.status),
                              )}
                            >
                              {server.status || "unknown"}
                            </Badge>
                          </>
                        )}
                      </div>
                      {server?.description && (
                        <p className="text-xs text-muted-foreground mt-1">
                          {server.description}
                        </p>
                      )}
                      {server?.url && (
                        <p className="text-xs text-muted-foreground mt-1">
                          {server.url}
                        </p>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="bg-blue-50 dark:bg-blue-950/30 border border-blue-200 dark:border-blue-900 rounded-lg p-3">
          <p className="text-sm text-blue-800 dark:text-blue-200">
            Total: <strong>{selectedServers.size}</strong> MCP server
            {selectedServers.size !== 1 ? "s" : ""} will be made public
          </p>
        </div>
      </div>
    );
  };

  const renderStepContent = () => {
    switch (currentStep) {
      case 0:
        return renderStep1Content();
      case 1:
        return renderStep2Content();
      default:
        return null;
    }
  };

  const renderStepButtons = () => {
    return (
      <div className="flex justify-between mt-6">
        <Button
          variant="outline"
          onClick={currentStep === 0 ? handleClose : handlePrevious}
        >
          {currentStep === 0 ? "Cancel" : "Previous"}
        </Button>

        <div className="flex space-x-2">
          {currentStep === 0 && (
            <Button onClick={handleNext} disabled={selectedServers.size === 0}>
              Next
            </Button>
          )}

          {currentStep === 1 && (
            <Button
              onClick={handleSubmit}
              disabled={loading}
              data-loading={loading ? "true" : undefined}
            >
              {loading ? "Making Public..." : "Make Public"}
            </Button>
          )}
        </div>
      </div>
    );
  };

  return (
    <Dialog
      open={visible}
      onOpenChange={(o) => (!o ? handleClose() : undefined)}
    >
      <DialogContent className="max-w-[1200px]">
        <DialogHeader>
          <DialogTitle>Make MCP Servers Public</DialogTitle>
        </DialogHeader>
        <Stepper current={currentStep} steps={["Select Servers", "Confirm"]} />
        {renderStepContent()}
        {renderStepButtons()}
      </DialogContent>
    </Dialog>
  );
};

export default MakeMCPPublicForm;
