import React, { useState, useEffect } from "react";
import { Button, message, Spin, Alert, Collapse, Badge } from "antd";
import { CheckCircleOutlined, ExclamationCircleOutlined, ReloadOutlined, ToolOutlined, InfoCircleOutlined } from "@ant-design/icons";
import { Card, Title, Text } from "@tremor/react";
import { testMCPToolsListRequest } from "../networking";

const { Panel } = Collapse;

interface MCPConnectionStatusProps {
  accessToken: string | null;
  formValues: Record<string, any>;
  onToolsLoaded?: (tools: any[]) => void;
}

const MCPConnectionStatus: React.FC<MCPConnectionStatusProps> = ({
  accessToken,
  formValues,
  onToolsLoaded
}) => {
  const [tools, setTools] = useState<any[]>([]);
  const [isLoadingTools, setIsLoadingTools] = useState(false);
  const [toolsError, setToolsError] = useState<string | null>(null);
  const [hasShownSuccessMessage, setHasShownSuccessMessage] = useState(false);

  // Check if we have the minimum required fields to fetch tools
  const canFetchTools = formValues.url && formValues.transport && formValues.auth_type && accessToken;

  const fetchTools = async () => {
    if (!accessToken || !formValues.url) {
      return;
    }

    setIsLoadingTools(true);
    setToolsError(null);
    
    try {
      // Prepare the MCP server config from form values
      const mcpServerConfig = {
        server_id: formValues.server_id || "",
        server_name: formValues.server_name || "",
        url: formValues.url,
        transport: formValues.transport,
        auth_type: formValues.auth_type,
        mcp_info: formValues.mcp_info,
      };

      const toolsResponse = await testMCPToolsListRequest(accessToken, mcpServerConfig);
      
      if (toolsResponse.tools && !toolsResponse.error) {
        setTools(toolsResponse.tools);
        setToolsError(null);
        onToolsLoaded?.(toolsResponse.tools);
        if (toolsResponse.tools.length > 0 && !hasShownSuccessMessage) {
          setHasShownSuccessMessage(true);
        }
      } else {
        const errorMessage = toolsResponse.message || "Failed to retrieve tools list";
        setToolsError(errorMessage);
        setTools([]);
        onToolsLoaded?.([]);
        setHasShownSuccessMessage(false);
      }
    } catch (error) {
      console.error("Tools fetch error:", error);
      setToolsError(error instanceof Error ? error.message : String(error));
      setTools([]);
      onToolsLoaded?.([]);
      setHasShownSuccessMessage(false);
    } finally {
      setIsLoadingTools(false);
    }
  };

  // Auto-fetch tools when form values change and required fields are available
  useEffect(() => {
    if (canFetchTools) {
      fetchTools();
    } else {
      // Clear tools if required fields are missing
      setTools([]);
      setToolsError(null);
      setHasShownSuccessMessage(false);
      onToolsLoaded?.([]);
    }
  }, [formValues.url, formValues.transport, formValues.auth_type, accessToken]);

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
            <Text className="text-sm">
              Fill in URL, Transport, and Authentication to test MCP server connection
            </Text>
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
                <Text className="text-gray-500 text-sm">
                  Server: {formValues.url}
                </Text>
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
                  <Button
                    icon={<ReloadOutlined />}
                    onClick={fetchTools}
                    size="small"
                  >
                    Retry
                  </Button>
                }
              />
            )}

            {!isLoadingTools && tools.length > 0 && (
              <Collapse
                items={[
                  {
                    key: '1',
                    label: (
                      <div className="flex items-center">
                        <ToolOutlined className="mr-2 text-green-500" />
                        <span className="font-medium">Available Tools</span>
                        <Badge 
                          count={tools.length} 
                          style={{ 
                            backgroundColor: '#52c41a', 
                            marginLeft: '8px' 
                          }} 
                        />
                      </div>
                    ),
                    children: (
                      <div className="space-y-2 max-h-48 overflow-y-auto">
                        {tools.map((tool, index) => (
                          <div key={index} className="p-3 bg-gray-50 rounded-lg">
                            <Text className="font-medium text-gray-900">{tool.name}</Text>
                            {tool.description && (
                              <Text className="text-gray-500 text-sm block mt-1">
                                {tool.description}
                              </Text>
                            )}
                          </div>
                        ))}
                      </div>
                    ),
                  },
                ]}
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