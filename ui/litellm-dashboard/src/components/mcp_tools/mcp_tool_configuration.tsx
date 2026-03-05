import React, { useEffect, useRef, useState } from "react";
import { Card, Title, Text } from "@tremor/react";
import { ToolOutlined, CheckCircleOutlined, SearchOutlined } from "@ant-design/icons";
import { Badge, Spin, Checkbox, Input } from "antd";
import { useTestMCPConnection } from "../../hooks/useTestMCPConnection";
import {
  categorizeTools,
  groupToolsByCategory,
  CRUD_CATEGORY_ORDER,
  CRUD_CATEGORY_COLORS,
} from "./tool_crud_categorization";

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
  const previousToolsRef = useRef<any[]>([]);
  const [toolSearchTerm, setToolSearchTerm] = useState("");
  const hasInitializedRef = useRef(false);

  const { tools, isLoadingTools, toolsError, canFetchTools } = useTestMCPConnection({
    accessToken,
    oauthAccessToken,
    formValues,
    enabled: true,
  });

  // Categorize and filter tools based on search term
  const categorizedTools = categorizeTools(tools);
  const filteredTools = categorizedTools.filter((tool) => {
    const searchLower = toolSearchTerm.toLowerCase();
    return (
      tool.name.toLowerCase().includes(searchLower) ||
      (tool.description && tool.description.toLowerCase().includes(searchLower))
    );
  });

  // Group filtered tools by category (always grouped)
  const groupedTools = groupToolsByCategory(filteredTools);

  // Auto-select tools when tools are first loaded or when tools list changes
  useEffect(() => {
    // Check if the tools list has actually changed by comparing tool names
    const currentToolNames = tools.map((tool) => tool.name).sort().join(",");
    const previousToolNames = previousToolsRef.current.map((tool) => tool.name).sort().join(",");
    const toolsListChanged = currentToolNames !== previousToolNames;

    if (tools.length > 0 && toolsListChanged) {
      const availableToolNames = tools.map((tool) => tool.name);
      
      // On initial load (first time tools are fetched)
      if (!hasInitializedRef.current) {
        hasInitializedRef.current = true;
        
        if (existingAllowedTools && existingAllowedTools.length > 0) {
          // Edit mode: pre-select tools that match existing allowed tools
          const validExistingTools = existingAllowedTools.filter((toolName) => availableToolNames.includes(toolName));
          onAllowedToolsChange(validExistingTools);
        } else {
          // Create mode: auto-select all tools
          onAllowedToolsChange(availableToolNames);
        }
      } else {
        // Tools list changed after initial load (e.g., URL was edited)
        // Keep any tools from the current selection that exist in the new tools list
        const matchingTools = allowedTools.filter((toolName) => availableToolNames.includes(toolName));
        onAllowedToolsChange(matchingTools);
      }
    } else if (tools.length === 0 && previousToolsRef.current.length > 0) {
      // Tools were cleared (e.g., URL became invalid or is being edited)
      // Don't clear allowedTools here - let the user keep their selection
      // until new tools are loaded
    }
    
    // Update ref to track current tools
    previousToolsRef.current = tools;
  }, [tools, allowedTools, existingAllowedTools, onAllowedToolsChange]);

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
  if (!canFetchTools && !formValues.url && !formValues.spec_path) {
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
        {!canFetchTools && (formValues.url || formValues.spec_path) && (
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

            {/* Search bar */}
            <Input
              placeholder="Search tools by name or description..."
              prefix={<SearchOutlined className="text-gray-400" />}
              value={toolSearchTerm}
              onChange={(e) => setToolSearchTerm(e.target.value)}
              allowClear
              className="rounded-lg"
              size="large"
            />

            {/* Tool list grouped by CRUD category */}
            <div className="space-y-2">
              {filteredTools.length === 0 ? (
                <div className="text-center py-6 text-gray-400 border rounded-lg border-dashed">
                  <SearchOutlined className="text-2xl mb-2" />
                  <Text>No tools found matching &quot;{toolSearchTerm}&quot;</Text>
                </div>
              ) : (
                CRUD_CATEGORY_ORDER.map((category) => {
                  const categoryTools = groupedTools[category];
                  if (categoryTools.length === 0) return null;

                  const categoryColors = CRUD_CATEGORY_COLORS[category];
                  const enabledCount = categoryTools.filter((tool) => allowedTools.includes(tool.name)).length;

                  return (
                    <div key={category} className="space-y-2">
                      <div className={`flex items-center gap-2 px-3 py-2 rounded-lg border ${categoryColors.bg} ${categoryColors.border}`}>
                        <Text className={`font-semibold ${categoryColors.text}`}>{category}</Text>
                        <Badge
                          count={categoryTools.length}
                          style={{ backgroundColor: category === "Create" ? "#52c41a" : category === "Read" ? "#1890ff" : category === "Update" ? "#faad14" : category === "Delete" ? "#ff4d4f" : "#8c8c8c" }}
                        />
                        <Text className={`text-xs ${categoryColors.text} ml-auto`}>
                          {enabledCount}/{categoryTools.length} enabled
                        </Text>
                      </div>
                      <div className="space-y-2 pl-4">
                        {categoryTools.map((tool, index) => (
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
                              <Checkbox
                                checked={allowedTools.includes(tool.name)}
                                onChange={() => handleToolToggle(tool.name)}
                              />
                              <div className="flex-1">
                                <div className="flex items-center gap-2">
                                  <Text className="font-medium text-gray-900">{tool.name}</Text>
                                  <span
                                    className={`px-2 py-0.5 text-xs rounded-full font-medium ${
                                      allowedTools.includes(tool.name)
                                        ? "bg-green-100 text-green-800"
                                        : "bg-red-100 text-red-800"
                                    }`}
                                  >
                                    {allowedTools.includes(tool.name) ? "Enabled" : "Disabled"}
                                  </span>
                                  <span className={`px-2 py-0.5 text-xs rounded-full font-medium ${categoryColors.badge}`}>
                                    {category}
                                  </span>
                                </div>
                                {tool.description && (
                                  <Text className="text-gray-500 text-sm block mt-1">{tool.description}</Text>
                                )}
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
                  );
                })
              )}
            </div>
          </div>
        )}
      </div>
    </Card>
  );
};

export default MCPToolConfiguration;
