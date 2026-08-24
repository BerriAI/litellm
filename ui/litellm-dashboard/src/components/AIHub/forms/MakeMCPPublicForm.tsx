import React, { useState, useEffect } from "react";
import { Loader2 } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { cn } from "@/lib/cva.config";
import { makeMCPPublicCall } from "../../networking";
import { toast } from "@/lib/toast";
import { MCPServerData } from "@/components/AIHub/MCPHubTableColumns";

const STEP_TITLES = ["Select Servers", "Confirm"];

const statusVariant = (status?: string) => {
  if (status === "active" || status === "healthy") {
    return "default" as const;
  }
  if (status === "inactive" || status === "unhealthy") {
    return "destructive" as const;
  }
  return "outline" as const;
};

interface MakeMCPPublicFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string;
  mcpHubData: MCPServerData[];
  onSuccess: () => void;
}

const MakeMCPPublicForm: React.FC<MakeMCPPublicFormProps> = ({
  visible,
  onClose,
  accessToken,
  mcpHubData,
  onSuccess,
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [selectedServers, setSelectedServers] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(false);

  const handleClose = () => {
    setCurrentStep(0);
    setSelectedServers(new Set());
    onClose();
  };

  const handleNext = () => {
    if (currentStep === 0) {
      if (selectedServers.size === 0) {
        toast.fromError("Please select at least one MCP server to make public");
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

  // Initialize and preselect already public servers when modal opens
  useEffect(() => {
    if (visible && mcpHubData.length > 0) {
      // Extract server IDs from servers that are already public
      const publicServerIds = mcpHubData
        .filter((server) => server.mcp_info?.is_public === true)
        .map((server) => server.server_id);

      // Preselect servers that are already public
      setSelectedServers(new Set(publicServerIds));
    }
  }, [visible]); // Only re-run when modal visibility changes, not when mcpHubData updates

  const handleSubmit = async () => {
    if (selectedServers.size === 0) {
      toast.fromError("Please select at least one MCP server to make public");
      return;
    }

    setLoading(true);
    try {
      const serverIdsToMakePublic = Array.from(selectedServers);

      // Make batch API call for all servers
      await makeMCPPublicCall(accessToken, serverIdsToMakePublic);

      toast.success(`Successfully made ${serverIdsToMakePublic.length} MCP server(s) public!`);
      handleClose();
      onSuccess();
    } catch (error) {
      console.error("Error making MCP servers public:", error);
      toast.fromError("Failed to make MCP servers public. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const renderStep1Content = () => {
    const allServersSelected =
      mcpHubData.length > 0 && mcpHubData.every((server) => selectedServers.has(server.server_id));
    const isIndeterminate = selectedServers.size > 0 && !allServersSelected;

    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold">Select MCP Servers to Make Public</h3>
          <div className="flex items-center space-x-2">
            <label className="flex items-center gap-2 text-sm">
              <Checkbox
                checked={allServersSelected}
                indeterminate={isIndeterminate}
                onCheckedChange={(checked) => handleSelectAll(checked === true)}
                disabled={mcpHubData.length === 0}
              />
              Select All {mcpHubData.length > 0 && `(${mcpHubData.length})`}
            </label>
          </div>
        </div>

        <p className="text-sm text-muted-foreground">
          Select the MCP servers you want to be visible on the public model hub. Users will still require a valid
          Virtual Key to use these servers.
        </p>

        <div className="max-h-96 overflow-y-auto border rounded-lg p-4">
          <div className="space-y-3">
            {mcpHubData.length === 0 ? (
              <div className="text-center py-8 text-muted-foreground">
                <p>No MCP servers available.</p>
              </div>
            ) : (
              mcpHubData.map((server) => {
                const isPublic = server.mcp_info?.is_public === true;
                return (
                  <div
                    key={server.server_id}
                    className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-accent"
                  >
                    <Checkbox
                      checked={selectedServers.has(server.server_id)}
                      onCheckedChange={(checked) => handleServerSelection(server.server_id, checked === true)}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="font-medium break-words">{server.server_name}</p>
                        {isPublic && <Badge>Public</Badge>}
                        <Badge variant="secondary">{server.transport}</Badge>
                        <Badge variant={statusVariant(server.status)}>{server.status || "unknown"}</Badge>
                      </div>
                      <p className="text-xs text-muted-foreground mt-1 break-words">
                        {server.description || server.url}
                      </p>
                      {server.allowed_tools && server.allowed_tools.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {server.allowed_tools.slice(0, 3).map((tool, idx) => (
                            <Badge key={idx} variant="outline">
                              {tool}
                            </Badge>
                          ))}
                          {server.allowed_tools.length > 3 && (
                            <p className="text-xs text-muted-foreground">+{server.allowed_tools.length - 3} more</p>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {selectedServers.size > 0 && (
          <div className="bg-info/10 border border-info/20 rounded-lg p-3">
            <p className="text-sm text-info">
              <strong>{selectedServers.size}</strong> MCP server{selectedServers.size !== 1 ? "s" : ""} selected
            </p>
          </div>
        )}
      </div>
    );
  };

  const renderStep2Content = () => {
    return (
      <div className="space-y-4">
        <h3 className="text-lg font-semibold">Confirm Making MCP Servers Public</h3>

        <div className="bg-warning/10 border border-warning/20 rounded-lg p-4">
          <p className="text-sm text-warning">
            <strong>Warning:</strong> Once you make these MCP servers public, anyone who can go to the{" "}
            <code>/ui/model_hub_table</code> will be able to know they exist on the proxy.
          </p>
        </div>

        <div className="space-y-3">
          <p className="font-medium">MCP Servers to be made public:</p>
          <div className="max-h-48 overflow-y-auto border rounded-lg p-3">
            <div className="space-y-2">
              {Array.from(selectedServers).map((serverId) => {
                const server = mcpHubData.find((s) => s.server_id === serverId);
                return (
                  <div key={serverId} className="flex items-center justify-between p-2 bg-muted rounded-sm">
                    <div className="flex-1 min-w-0">
                      <div className="flex flex-wrap items-center gap-2">
                        <p className="font-medium break-words">{server?.server_name || serverId}</p>
                        {server && (
                          <>
                            <Badge variant="secondary">{server.transport}</Badge>
                            <Badge variant={statusVariant(server.status)}>{server.status || "unknown"}</Badge>
                          </>
                        )}
                      </div>
                      {server?.description && (
                        <p className="text-xs text-muted-foreground mt-1 break-words">{server.description}</p>
                      )}
                      {server?.url && <p className="text-xs text-muted-foreground mt-1 break-words">{server.url}</p>}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="bg-info/10 border border-info/20 rounded-lg p-3">
          <p className="text-sm text-info">
            Total: <strong>{selectedServers.size}</strong> MCP server{selectedServers.size !== 1 ? "s" : ""} will be
            made public
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
        <Button variant="outline" onClick={currentStep === 0 ? handleClose : handlePrevious}>
          {currentStep === 0 ? "Cancel" : "Previous"}
        </Button>

        <div className="flex space-x-2">
          {currentStep === 0 && (
            <Button onClick={handleNext} disabled={selectedServers.size === 0}>
              Next
            </Button>
          )}

          {currentStep === 1 && (
            <Button onClick={handleSubmit} disabled={loading}>
              {loading && <Loader2 className="size-4 animate-spin" />}
              Make Public
            </Button>
          )}
        </div>
      </div>
    );
  };

  return (
    <Dialog open={visible} onOpenChange={(open) => !open && handleClose()} disablePointerDismissal>
      <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1200px]">
        <DialogHeader>
          <DialogTitle>Make MCP Servers Public</DialogTitle>
        </DialogHeader>

        <div>
          <ol className="mb-6 flex items-center gap-6">
            {STEP_TITLES.map((title, index) => (
              <li
                key={title}
                className="flex items-center gap-2"
                aria-current={currentStep === index ? "step" : undefined}
              >
                <span
                  className={cn(
                    "flex size-6 items-center justify-center rounded-full border text-xs",
                    currentStep === index
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-border text-muted-foreground",
                  )}
                >
                  {index + 1}
                </span>
                <span className={cn("text-sm", currentStep === index ? "font-medium" : "text-muted-foreground")}>
                  {title}
                </span>
              </li>
            ))}
          </ol>

          {renderStepContent()}
          {renderStepButtons()}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default MakeMCPPublicForm;
