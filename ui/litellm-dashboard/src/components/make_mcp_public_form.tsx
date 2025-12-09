import React, { useState, useEffect } from "react";
import { Modal, Form, Steps, Button, Checkbox } from "antd";
import { Text, Title, Badge } from "@tremor/react";
import { makeMCPPublicCall } from "./networking";
import NotificationsManager from "./molecules/notifications_manager";
import { MCPServerData } from "./mcp_hub_table_columns";

const { Step } = Steps;

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
  const [form] = Form.useForm();

  const handleClose = () => {
    setCurrentStep(0);
    setSelectedServers(new Set());
    form.resetFields();
    onClose();
  };

  const handleNext = () => {
    if (currentStep === 0) {
      if (selectedServers.size === 0) {
        NotificationsManager.fromBackend("Please select at least one MCP server to make public");
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
      NotificationsManager.fromBackend("Please select at least one MCP server to make public");
      return;
    }

    setLoading(true);
    try {
      const serverIdsToMakePublic = Array.from(selectedServers);
      
      // Make batch API call for all servers
      await makeMCPPublicCall(accessToken, serverIdsToMakePublic);

      NotificationsManager.success(`Successfully made ${serverIdsToMakePublic.length} MCP server(s) public!`);
      handleClose();
      onSuccess();
    } catch (error) {
      console.error("Error making MCP servers public:", error);
      NotificationsManager.fromBackend("Failed to make MCP servers public. Please try again.");
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
          <Title>Select MCP Servers to Make Public</Title>
          <div className="flex items-center space-x-2">
            <Checkbox
              checked={allServersSelected}
              indeterminate={isIndeterminate}
              onChange={(e) => handleSelectAll(e.target.checked)}
              disabled={mcpHubData.length === 0}
            >
              Select All {mcpHubData.length > 0 && `(${mcpHubData.length})`}
            </Checkbox>
          </div>
        </div>

        <Text className="text-sm text-gray-600">
          Select the MCP servers you want to be visible on the public model hub. Users will still require a valid API key to
          use these servers.
        </Text>

        <div className="max-h-96 overflow-y-auto border rounded-lg p-4">
          <div className="space-y-3">
            {mcpHubData.length === 0 ? (
              <div className="text-center py-8 text-gray-500">
                <Text>No MCP servers available.</Text>
              </div>
            ) : (
              mcpHubData.map((server) => {
                const isPublic = server.mcp_info?.is_public === true;
                return (
                  <div
                    key={server.server_id}
                    className="flex items-center space-x-3 p-3 border rounded-lg hover:bg-gray-50"
                  >
                    <Checkbox
                      checked={selectedServers.has(server.server_id)}
                      onChange={(e) => handleServerSelection(server.server_id, e.target.checked)}
                    />
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <Text className="font-medium">{server.server_name}</Text>
                        {isPublic && (
                          <Badge color="emerald" size="sm">
                            Public
                          </Badge>
                        )}
                        <Badge color="blue" size="sm">
                          {server.transport}
                        </Badge>
                        <Badge 
                          color={
                            server.status === "active" || server.status === "healthy" 
                              ? "green" 
                              : server.status === "inactive" || server.status === "unhealthy"
                              ? "red"
                              : "gray"
                          } 
                          size="sm"
                        >
                          {server.status || "unknown"}
                        </Badge>
                      </div>
                      <Text className="text-xs text-gray-600 mt-1">
                        {server.description || server.url}
                      </Text>
                      {server.allowed_tools && server.allowed_tools.length > 0 && (
                        <div className="flex flex-wrap gap-1 mt-1">
                          {server.allowed_tools.slice(0, 3).map((tool, idx) => (
                            <Badge key={idx} color="purple" size="xs">
                              {tool}
                            </Badge>
                          ))}
                          {server.allowed_tools.length > 3 && (
                            <Text className="text-xs text-gray-500">+{server.allowed_tools.length - 3} more</Text>
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
          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <Text className="text-sm text-blue-800">
              <strong>{selectedServers.size}</strong> MCP server{selectedServers.size !== 1 ? "s" : ""} selected
            </Text>
          </div>
        )}
      </div>
    );
  };

  const renderStep2Content = () => {
    return (
      <div className="space-y-4">
        <Title>Confirm Making MCP Servers Public</Title>

        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <Text className="text-sm text-yellow-800">
            <strong>Warning:</strong> Once you make these MCP servers public, anyone who can go to the{" "}
            <code>/ui/model_hub_table</code> will be able to know they exist on the proxy.
          </Text>
        </div>

        <div className="space-y-3">
          <Text className="font-medium">MCP Servers to be made public:</Text>
          <div className="max-h-48 overflow-y-auto border rounded-lg p-3">
            <div className="space-y-2">
              {Array.from(selectedServers).map((serverId) => {
                const server = mcpHubData.find((s) => s.server_id === serverId);
                return (
                  <div key={serverId} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <Text className="font-medium">{server?.server_name || serverId}</Text>
                        {server && (
                          <>
                            <Badge color="blue" size="xs">
                              {server.transport}
                            </Badge>
                            <Badge 
                              color={
                                server.status === "active" || server.status === "healthy" 
                                  ? "green" 
                                  : server.status === "inactive" || server.status === "unhealthy"
                                  ? "red"
                                  : "gray"
                              } 
                              size="xs"
                            >
                              {server.status || "unknown"}
                            </Badge>
                          </>
                        )}
                      </div>
                      {server?.description && (
                        <Text className="text-xs text-gray-600 mt-1">{server.description}</Text>
                      )}
                      {server?.url && (
                        <Text className="text-xs text-gray-500 mt-1">{server.url}</Text>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>

        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <Text className="text-sm text-blue-800">
            Total: <strong>{selectedServers.size}</strong> MCP server{selectedServers.size !== 1 ? "s" : ""} will be made
            public
          </Text>
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
        <Button onClick={currentStep === 0 ? handleClose : handlePrevious}>
          {currentStep === 0 ? "Cancel" : "Previous"}
        </Button>

        <div className="flex space-x-2">
          {currentStep === 0 && (
            <Button onClick={handleNext} disabled={selectedServers.size === 0}>
              Next
            </Button>
          )}

          {currentStep === 1 && (
            <Button onClick={handleSubmit} loading={loading}>
              Make Public
            </Button>
          )}
        </div>
      </div>
    );
  };

  return (
    <Modal
      title="Make MCP Servers Public"
      open={visible}
      onCancel={handleClose}
      footer={null}
      width={1200}
      maskClosable={false}
    >
      <Form form={form} layout="vertical">
        <Steps current={currentStep} className="mb-6">
          <Step title="Select Servers" />
          <Step title="Confirm" />
        </Steps>

        {renderStepContent()}
        {renderStepButtons()}
      </Form>
    </Modal>
  );
};

export default MakeMCPPublicForm;

