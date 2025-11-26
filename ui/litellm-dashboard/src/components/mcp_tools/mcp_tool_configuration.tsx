import React, { useEffect, useRef } from "react";
import { Card, Title, Text } from "@tremor/react";
import { ToolOutlined, CheckCircleOutlined } from "@ant-design/icons";
import { Badge, Spin, Checkbox } from "antd";
import { useTestMCPConnection } from "../../hooks/useTestMCPConnection";

interface MCPToolConfigurationProps {
  accessToken: string | null;
  oauthAccessToken?: string | null;
  formValues: Record<string, any>;
  allowedTools: string[];
  existingAllowedTools: string[] | null;
  onAllowedToolsChange: (tools: string[]) => void;
}

const MCPToolConfiguration: React.FC<MCPToolConfigurationProps> = ({
  accessToken,
  oauthAccessToken,
  formValues,
  allowedTools,
  existingAllowedTools,
  onAllowedToolsChange,
}) => {
  const previousToolsLengthRef = useRef(0);

  const { tools, isLoadingTools, toolsError, canFetchTools } = useTestMCPConnection({
    accessToken,
    oauthAccessToken,
    formValues,
    enabled: true,
  });

  // Auto-select tools when tools are first loaded
  useEffect(() => {
    // Only auto-select if:
    // 1. We have tools
    // 2. Tools length changed (new tools loaded)
    // 3. No tools are currently selected (initial state)
    if (tools.length > 0 && tools.length !== previousToolsLengthRef.current && allowedTools.length === 0) {
      if (existingAllowedTools && existingAllowedTools.length > 0) {
        // If we have existing allowed tools, use those as the initial selection
        // Filter to only include tools that are actually available from the server
        const availableToolNames = tools.map((tool) => tool.name);
        const validExistingTools = existingAllowedTools.filter((toolName) => availableToolNames.includes(toolName));
        onAllowedToolsChange(validExistingTools);
      } else {
        // If no existing allowed tools, auto-select all tools (create mode)
        const allToolNames = tools.map((tool) => tool.name);
        onAllowedToolsChange(allToolNames);
      }
    }
    // Update ref to track tools length (will be 0 when tools clear)
    previousToolsLengthRef.current = tools.length;
  }, [tools, allowedTools.length, existingAllowedTools, onAllowedToolsChange]);

  const handleToolToggle = (toolName: string) => {
    if (allowedTools.includes(toolName)) {
      onAllowedToolsChange(allowedTools.filter((name) => name !== toolName));
    } else {
      onAllowedToolsChange([...allowedTools, toolName]);
    }
  };

  const handleSelectAll = () => {
    const allToolNames = tools.map((tool) => tool.name);
    onAllowedToolsChange(allToolNames);
  };

  const handleDeselectAll = () => {
    onAllowedToolsChange([]);
  };

  // Don't show anything if required fields aren't filled
  if (!canFetchTools && !formValues.url) {
    return null;
  }

  return (
    <Card>
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ToolOutlined className="text-blue-600" />
            <Title>Tool Configuration</Title>
            {tools.length > 0 && (
              <Badge
                count={tools.length}
                style={{
                  backgroundColor: "#52c41a",
                }}
              />
            )}
          </div>
        </div>

        {/* Description */}
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
          <Text className="text-blue-800 text-sm">
            <strong>Select which tools users can call:</strong> Only checked tools will be available for users to
            invoke. Unchecked tools will be blocked from execution.
          </Text>
        </div>

        {/* Loading state */}
        {isLoadingTools && (
          <div className="flex items-center justify-center py-6">
            <Spin size="large" />
            <Text className="ml-3">Loading tools...</Text>
          </div>
        )}

        {/* Error state */}
        {toolsError && !isLoadingTools && (
          <div className="text-center py-6 text-red-500 border rounded-lg border-dashed border-red-300 bg-red-50">
            <ToolOutlined className="text-2xl mb-2" />
            <Text className="text-red-600 font-medium">Unable to load tools</Text>
            <br />
            <Text className="text-sm text-red-500">{toolsError}</Text>
          </div>
        )}

        {/* No tools state */}
        {!isLoadingTools && !toolsError && tools.length === 0 && canFetchTools && (
          <div className="text-center py-6 text-gray-400 border rounded-lg border-dashed">
            <ToolOutlined className="text-2xl mb-2" />
            <Text>No tools available for configuration</Text>
            <br />
            <Text className="text-sm">Connect to an MCP server with tools to configure them</Text>
          </div>
        )}

        {/* Incomplete form state */}
        {!canFetchTools && formValues.url && (
          <div className="text-center py-6 text-gray-400 border rounded-lg border-dashed">
            <ToolOutlined className="text-2xl mb-2" />
            <Text>Complete required fields to configure tools</Text>
            <br />
            <Text className="text-sm">Fill in URL, Transport, and Authentication to load available tools</Text>
          </div>
        )}

        {/* Tools loaded successfully */}
        {!isLoadingTools && !toolsError && tools.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 p-3 bg-green-50 rounded-lg border border-green-200 flex-1">
                <CheckCircleOutlined className="text-green-600" />
                <Text className="text-green-700 font-medium">
                  {allowedTools.length} of {tools.length} {tools.length === 1 ? "tool" : "tools"} enabled for user
                  access
                </Text>
              </div>
              <div className="flex gap-2 ml-3">
                <button
                  type="button"
                  onClick={handleSelectAll}
                  className="px-3 py-1.5 text-sm text-blue-600 hover:text-blue-700 hover:bg-blue-50 rounded-md transition-colors"
                >
                  Enable All
                </button>
                <button
                  type="button"
                  onClick={handleDeselectAll}
                  className="px-3 py-1.5 text-sm text-gray-600 hover:text-gray-700 hover:bg-gray-100 rounded-md transition-colors"
                >
                  Disable All
                </button>
              </div>
            </div>

            {/* Tool list with checkboxes */}
            <div className="space-y-2">
              {tools.map((tool, index) => (
                <div
                  key={index}
                  className={`p-4 rounded-lg border transition-colors cursor-pointer ${
                    allowedTools.includes(tool.name)
                      ? "bg-blue-50 border-blue-300 hover:border-blue-400"
                      : "bg-gray-50 border-gray-200 hover:border-gray-300"
                  }`}
                  onClick={() => handleToolToggle(tool.name)}
                >
                  <div className="flex items-start gap-3">
                    <Checkbox checked={allowedTools.includes(tool.name)} onChange={() => handleToolToggle(tool.name)} />
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <Text className="font-medium text-gray-900">{tool.name}</Text>
                        <span
                          className={`px-2 py-0.5 text-xs rounded-full font-medium ${
                            allowedTools.includes(tool.name) ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
                          }`}
                        >
                          {allowedTools.includes(tool.name) ? "Enabled" : "Disabled"}
                        </span>
                      </div>
                      {tool.description && <Text className="text-gray-500 text-sm block mt-1">{tool.description}</Text>}
                      <Text className="text-gray-400 text-xs block mt-1">
                        {allowedTools.includes(tool.name)
                          ? "✓ Users can call this tool"
                          : "✗ Users cannot call this tool"}
                      </Text>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default MCPToolConfiguration;
