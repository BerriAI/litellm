import React, { useEffect } from "react";
import { Button, Spin, Alert } from "antd";
import { CheckCircleOutlined, ExclamationCircleOutlined, ReloadOutlined, ToolOutlined } from "@ant-design/icons";
import { Card, Title, Text } from "@tremor/react";
import { useTestMCPConnection } from "../../hooks/useTestMCPConnection";

interface MCPConnectionStatusProps {
  accessToken: string | null;
  formValues: Record<string, any>;
  onToolsLoaded?: (tools: any[]) => void;
}

const MCPConnectionStatus: React.FC<MCPConnectionStatusProps> = ({ accessToken, formValues, onToolsLoaded }) => {
  const { tools, isLoadingTools, toolsError, canFetchTools, fetchTools } = useTestMCPConnection({
    accessToken,
    formValues,
    enabled: true, // Auto-fetch when required fields are available
  });

  // Notify parent component when tools change
  useEffect(() => {
    onToolsLoaded?.(tools);
  }, [tools, onToolsLoaded]);

  // Don't show anything if required fields aren't filled
  if (!canFetchTools && !formValues.url) {
    return null;
  }

  return (
    <Card>
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <CheckCircleOutlined className="text-blue-600" />
          <Title>Connection Status</Title>
        </div>

        {!canFetchTools && formValues.url && (
          <div className="text-center py-6 text-gray-400 border rounded-lg border-dashed">
            <ToolOutlined className="text-2xl mb-2" />
            <Text>Complete required fields to test connection</Text>
            <br />
            <Text className="text-sm">Fill in URL, Transport, and Authentication to test MCP server connection</Text>
          </div>
        )}

        {canFetchTools && (
          <div>
            <div className="flex items-center justify-between mb-4">
              <div>
                <Text className="text-gray-700 font-medium">
                  {isLoadingTools
                    ? "Testing connection to MCP server..."
                    : tools.length > 0
                      ? "Connection successful"
                      : toolsError
                        ? "Connection failed"
                        : "Ready to test connection"}
                </Text>
                <br />
                <Text className="text-gray-500 text-sm">Server: {formValues.url}</Text>
              </div>

              {isLoadingTools && (
                <div className="flex items-center text-blue-600">
                  <Spin size="small" className="mr-2" />
                  <Text className="text-blue-600">Connecting...</Text>
                </div>
              )}

              {!isLoadingTools && !toolsError && tools.length > 0 && (
                <div className="flex items-center text-green-600">
                  <CheckCircleOutlined className="mr-1" />
                  <Text className="text-green-600 font-medium">Connected</Text>
                </div>
              )}

              {toolsError && (
                <div className="flex items-center text-red-600">
                  <ExclamationCircleOutlined className="mr-1" />
                  <Text className="text-red-600 font-medium">Failed</Text>
                </div>
              )}
            </div>

            {isLoadingTools && (
              <div className="flex items-center justify-center py-6">
                <Spin size="large" />
                <Text className="ml-3">Testing connection and loading tools...</Text>
              </div>
            )}

            {toolsError && (
              <Alert
                message="Connection Failed"
                description={toolsError}
                type="error"
                showIcon
                action={
                  <Button icon={<ReloadOutlined />} onClick={fetchTools} size="small">
                    Retry
                  </Button>
                }
              />
            )}

            {!isLoadingTools && tools.length === 0 && !toolsError && (
              <div className="text-center py-6 text-gray-500 border rounded-lg border-dashed">
                <CheckCircleOutlined className="text-2xl mb-2 text-green-500" />
                <Text className="text-green-600 font-medium">Connection successful!</Text>
                <br />
                <Text className="text-gray-500">No tools found for this MCP server</Text>
              </div>
            )}
          </div>
        )}
      </div>
    </Card>
  );
};

export default MCPConnectionStatus;
