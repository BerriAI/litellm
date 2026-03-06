import React, { useEffect, useRef, useState } from "react";
import { Card, Title, Text } from "@tremor/react";
import { ToolOutlined, CheckCircleOutlined, SearchOutlined, EditOutlined } from "@ant-design/icons";
import { Badge, Spin, Checkbox, Input } from "antd";
import { useTestMCPConnection } from "../../hooks/useTestMCPConnection";

interface MCPToolConfigurationProps {
  accessToken: string | null;
  oauthAccessToken?: string | null;
  formValues: Record<string, any>;
  allowedTools: string[];
  existingAllowedTools: string[] | null;
  onAllowedToolsChange: (tools: string[]) => void;
  toolNameToDisplayName: Record<string, string>;
  toolNameToDescription: Record<string, string>;
  onToolNameToDisplayNameChange: (map: Record<string, string>) => void;
  onToolNameToDescriptionChange: (map: Record<string, string>) => void;
}

const MCPToolConfiguration: React.FC<MCPToolConfigurationProps> = ({
  accessToken,
  oauthAccessToken,
  formValues,
  allowedTools,
  existingAllowedTools,
  onAllowedToolsChange,
  toolNameToDisplayName,
  toolNameToDescription,
  onToolNameToDisplayNameChange,
  onToolNameToDescriptionChange,
}) => {
  const previousToolsRef = useRef<any[]>([]);
  const [toolSearchTerm, setToolSearchTerm] = useState("");
  const hasInitializedRef = useRef(false);
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());

  const { tools, isLoadingTools, toolsError, canFetchTools } = useTestMCPConnection({
    accessToken,
    oauthAccessToken,
    formValues,
    enabled: true,
  });

  // Filter tools based on search term
  const filteredTools = tools.filter((tool) => {
    const searchLower = toolSearchTerm.toLowerCase();
    return (
      tool.name.toLowerCase().includes(searchLower) ||
      (tool.description && tool.description.toLowerCase().includes(searchLower))
    );
  });

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

  const handleToggleEditExpanded = (toolName: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setExpandedTools((prev) => {
      const next = new Set(prev);
      if (next.has(toolName)) {
        next.delete(toolName);
      } else {
        next.add(toolName);
      }
      return next;
    });
  };

  const handleDisplayNameChange = (toolName: string, value: string) => {
    const next = { ...toolNameToDisplayName };
    if (value) {
      next[toolName] = value;
    } else {
      delete next[toolName];
    }
    onToolNameToDisplayNameChange(next);
  };

  const handleDescriptionChange = (toolName: string, value: string) => {
    const next = { ...toolNameToDescription };
    if (value) {
      next[toolName] = value;
    } else {
      delete next[toolName];
    }
    onToolNameToDescriptionChange(next);
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

            {/* Tool list with checkboxes */}
            <div className="space-y-2">
              {filteredTools.length === 0 ? (
                <div className="text-center py-6 text-gray-400 border rounded-lg border-dashed">
                  <SearchOutlined className="text-2xl mb-2" />
                  <Text>No tools found matching &quot;{toolSearchTerm}&quot;</Text>
                </div>
              ) : (
                filteredTools.map((tool, index) => {
                  const isEnabled = allowedTools.includes(tool.name);
                  const isEditExpanded = expandedTools.has(tool.name);
                  return (
                  <div
                    key={index}
                    className={`rounded-lg border transition-colors ${
                      isEnabled
                        ? "bg-blue-50 border-blue-300 hover:border-blue-400"
                        : "bg-gray-50 border-gray-200 hover:border-gray-300"
                    }`}
                  >
                    {/* Main tool row */}
                    <div
                      className="p-4 cursor-pointer"
                      onClick={() => handleToolToggle(tool.name)}
                    >
                      <div className="flex items-start gap-3">
                        <Checkbox checked={isEnabled} onChange={() => handleToolToggle(tool.name)} />
                        <div className="flex-1">
                          <div className="flex items-center gap-2">
                            <Text className="font-medium text-gray-900">
                              {toolNameToDisplayName[tool.name] || tool.name}
                            </Text>
                            <span
                              className={`px-2 py-0.5 text-xs rounded-full font-medium ${
                                isEnabled ? "bg-green-100 text-green-800" : "bg-red-100 text-red-800"
                              }`}
                            >
                              {isEnabled ? "Enabled" : "Disabled"}
                            </span>
                            {toolNameToDisplayName[tool.name] && (
                              <span className="px-2 py-0.5 text-xs rounded-full font-medium bg-purple-100 text-purple-800">
                                Custom name
                              </span>
                            )}
                          </div>
                          {(toolNameToDescription[tool.name] || tool.description) && (
                            <Text className="text-gray-500 text-sm block mt-1">
                              {toolNameToDescription[tool.name] || tool.description}
                            </Text>
                          )}
                          <Text className="text-gray-400 text-xs block mt-1">
                            {isEnabled ? "✓ Users can call this tool" : "✗ Users cannot call this tool"}
                          </Text>
                        </div>
                        {/* Edit toggle button */}
                        <button
                          type="button"
                          onClick={(e) => handleToggleEditExpanded(tool.name, e)}
                          className={`p-1.5 rounded-md transition-colors ${
                            isEditExpanded
                              ? "bg-blue-100 text-blue-600"
                              : "text-gray-400 hover:text-gray-600 hover:bg-gray-100"
                          }`}
                          title="Edit display name and description"
                        >
                          <EditOutlined />
                        </button>
                      </div>
                    </div>

                    {/* Inline edit section */}
                    {isEditExpanded && (
                      <div
                        className="px-4 pb-4 pt-3 border-t border-gray-200 space-y-3 bg-gray-50 rounded-b-lg"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <div>
                          <Text className="text-xs font-medium text-gray-600 mb-1 block">
                            Display Name
                          </Text>
                          <Input
                            placeholder={tool.name}
                            value={toolNameToDisplayName[tool.name] || ""}
                            onChange={(e) => handleDisplayNameChange(tool.name, e.target.value)}
                          />
                          <Text className="text-xs text-gray-400 mt-1 block">
                            Override how this tool&apos;s name appears to users. Leave blank to use original.
                          </Text>
                        </div>
                        <div>
                          <Text className="text-xs font-medium text-gray-600 mb-1 block">
                            Description
                          </Text>
                          <Input.TextArea
                            placeholder={tool.description || "No description"}
                            value={toolNameToDescription[tool.name] || ""}
                            onChange={(e) => handleDescriptionChange(tool.name, e.target.value)}
                            rows={2}
                          />
                          <Text className="text-xs text-gray-400 mt-1 block">
                            Override the tool description shown to users. Leave blank to use original.
                          </Text>
                        </div>
                      </div>
                    )}
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
