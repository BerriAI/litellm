import React, { useEffect, useMemo, useRef, useState } from "react";
import { Card, Title, Text } from "@tremor/react";
import { ToolOutlined, CheckCircleOutlined, SearchOutlined, EditOutlined } from "@ant-design/icons";
import { Badge, Spin, Checkbox, Input, Radio } from "antd";
import { useTestMCPConnection } from "../../hooks/useTestMCPConnection";
import McpCrudPermissionPanel from "./McpCrudPermissionPanel";

interface KeyTool {
  name: string;
  description: string;
}

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
  /** Curated key tools from the OpenAPI registry preset (shown before spec loads). */
  keyTools?: KeyTool[];
  /** External tool state lifted from parent to avoid duplicate fetch requests. */
  externalTools?: any[];
  externalIsLoading?: boolean;
  externalError?: string | null;
  externalCanFetch?: boolean;
}

interface ToolEntry {
  name: string;
  description?: string;
}

interface ToolRowProps {
  tool: ToolEntry;
  isEnabled: boolean;
  isEditExpanded: boolean;
  toolNameToDisplayName: Record<string, string>;
  toolNameToDescription: Record<string, string>;
  onToggle: (name: string) => void;
  onToggleExpand: (name: string, e: React.MouseEvent) => void;
  onDisplayNameChange: (name: string, value: string) => void;
  onDescriptionChange: (name: string, value: string) => void;
}

const ToolRow: React.FC<ToolRowProps> = ({
  tool,
  isEnabled,
  isEditExpanded,
  toolNameToDisplayName,
  toolNameToDescription,
  onToggle,
  onToggleExpand,
  onDisplayNameChange,
  onDescriptionChange,
}) => (
  <div
    className={`rounded-lg border transition-colors ${
      isEnabled
        ? "bg-blue-50 border-blue-300 hover:border-blue-400"
        : "bg-gray-50 border-gray-200 hover:border-gray-300"
    }`}
  >
    <div className="p-4 cursor-pointer" onClick={() => onToggle(tool.name)}>
      <div className="flex items-start gap-3">
        <Checkbox checked={isEnabled} onChange={() => onToggle(tool.name)} />
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
        <button
          type="button"
          onClick={(e) => onToggleExpand(tool.name, e)}
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
    {isEditExpanded && (
      <div
        className="px-4 pb-4 pt-3 border-t border-gray-200 space-y-3 bg-gray-50 rounded-b-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <Text className="text-xs font-medium text-gray-600 mb-1 block">Display Name</Text>
          <Input
            placeholder={tool.name}
            value={toolNameToDisplayName[tool.name] || ""}
            onChange={(e) => onDisplayNameChange(tool.name, e.target.value)}
          />
          <Text className="text-xs text-gray-400 mt-1 block">
            Override how this tool&apos;s name appears to users. Leave blank to use original.
          </Text>
        </div>
        <div>
          <Text className="text-xs font-medium text-gray-600 mb-1 block">Description</Text>
          <Input.TextArea
            placeholder={tool.description || "No description"}
            value={toolNameToDescription[tool.name] || ""}
            onChange={(e) => onDescriptionChange(tool.name, e.target.value)}
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
  keyTools,
  externalTools,
  externalIsLoading,
  externalError,
  externalCanFetch,
}) => {
  const previousToolsRef = useRef<ToolEntry[]>([]);
  const [toolSearchTerm, setToolSearchTerm] = useState("");
  const [viewMode, setViewMode] = useState<"crud" | "flat">("crud");
  const hasInitializedRef = useRef(false);
  const previousSuggestedToolNamesRef = useRef<string>("");
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());

  // Use external tool state when provided (avoids duplicate fetch with MCPConnectionStatus).
  // Fall back to internal hook when used standalone (e.g., edit flow).
  const hasExternalState = externalTools !== undefined;
  const internalHook = useTestMCPConnection({
    accessToken,
    oauthAccessToken,
    formValues,
    enabled: !hasExternalState,
  });
  const tools: ToolEntry[] = hasExternalState ? externalTools : internalHook.tools;
  const isLoadingTools = hasExternalState ? (externalIsLoading ?? false) : internalHook.isLoadingTools;
  const toolsError = hasExternalState ? (externalError ?? null) : internalHook.toolsError;
  const canFetchTools = hasExternalState ? (externalCanFetch ?? false) : internalHook.canFetchTools;

  // Fuzzy-match curated key tool names against actual loaded tool names
  const suggestedTools = useMemo(() => {
    if (!keyTools || keyTools.length === 0 || tools.length === 0) return [];
    const usedNames = new Set<string>();
    const result: typeof tools = [];
    for (const keyTool of keyTools) {
      const keywords = keyTool.name.split("_").map((k) => k.toLowerCase()).filter((k) => k.length > 1);
      if (keywords.length === 0) continue;
      const normalize = (s: string) => s.toLowerCase().replace(/[-_/]/g, " ");
      let match = tools.find((t) => {
        if (usedNames.has(t.name)) return false;
        const n = normalize(t.name);
        return keywords.every((kw) => n.includes(kw));
      });
      if (!match) {
        const mainKw = keywords.find((k) => k.length > 3) ?? keywords[keywords.length - 1];
        match = tools.find((t) => {
          if (usedNames.has(t.name)) return false;
          return normalize(t.name).includes(mainKw);
        });
      }
      if (match) {
        result.push(match);
        usedNames.add(match.name);
      }
    }
    return result;
  }, [keyTools, tools]);

  const suggestedToolNames = useMemo(
    () => new Set(suggestedTools.map((t) => t.name)),
    [suggestedTools]
  );

  // Filter tools based on search term
  const filteredTools = useMemo(
    () =>
      tools.filter((tool) => {
        const searchLower = toolSearchTerm.toLowerCase();
        return (
          tool.name.toLowerCase().includes(searchLower) ||
          (tool.description && tool.description.toLowerCase().includes(searchLower))
        );
      }),
    [tools, toolSearchTerm]
  );

  const pinnedFiltered = useMemo(
    () => filteredTools.filter((t) => suggestedToolNames.has(t.name)),
    [filteredTools, suggestedToolNames]
  );

  const restFiltered = useMemo(
    () => filteredTools.filter((t) => !suggestedToolNames.has(t.name)),
    [filteredTools, suggestedToolNames]
  );

  // Auto-select tools when tools are first loaded or when tools list changes
  useEffect(() => {
    const currentToolNames = tools.map((tool) => tool.name).sort().join(",");
    const previousToolNames = previousToolsRef.current.map((tool) => tool.name).sort().join(",");
    const toolsListChanged = currentToolNames !== previousToolNames;

    // Reset initialization when a new preset is selected (suggestedTools fingerprint changes)
    const currentSuggestedNames = suggestedTools.map((t) => t.name).sort().join(",");
    if (currentSuggestedNames !== previousSuggestedToolNamesRef.current) {
      previousSuggestedToolNamesRef.current = currentSuggestedNames;
      if (currentSuggestedNames !== "") {
        hasInitializedRef.current = false;
      }
    }

    if (tools.length > 0 && toolsListChanged) {
      const availableToolNames = tools.map((tool) => tool.name);

      if (!hasInitializedRef.current) {
        hasInitializedRef.current = true;

        if (existingAllowedTools && existingAllowedTools.length > 0) {
          // Edit mode: pre-select tools that match existing allowed tools
          const validExistingTools = existingAllowedTools.filter((toolName) =>
            availableToolNames.includes(toolName)
          );
          onAllowedToolsChange(validExistingTools);
        } else if (suggestedTools.length > 0) {
          // OpenAPI preset: only enable suggested tools by default
          onAllowedToolsChange(
            suggestedTools.map((t) => t.name).filter((name) => availableToolNames.includes(name))
          );
        } else {
          // Create mode: auto-select all tools
          onAllowedToolsChange(availableToolNames);
        }
      } else {
        // Tools list changed after initial load (e.g., URL was edited)
        const matchingTools = allowedTools.filter((toolName) => availableToolNames.includes(toolName));
        onAllowedToolsChange(matchingTools);
      }
    }

    previousToolsRef.current = tools;
  }, [tools, allowedTools, existingAllowedTools, onAllowedToolsChange, suggestedTools]);

  const handleToolToggle = (toolName: string) => {
    if (allowedTools.includes(toolName)) {
      onAllowedToolsChange(allowedTools.filter((name) => name !== toolName));
    } else {
      onAllowedToolsChange([...allowedTools, toolName]);
    }
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

  const handleEnableSuggested = () => {
    // Enable ALL suggested tools (not just the currently filtered subset)
    const suggestedNames = suggestedTools.map((t) => t.name);
    const others = allowedTools.filter((n) => !suggestedToolNames.has(n));
    onAllowedToolsChange([...others, ...suggestedNames]);
  };

  const handleDisableSuggested = () => {
    // Disable ALL suggested tools (not just the currently filtered subset)
    onAllowedToolsChange(allowedTools.filter((n) => !suggestedToolNames.has(n)));
  };

  const handleEnableRest = () => {
    // Enable ALL non-suggested tools (not just the currently filtered subset)
    const restNames = tools.filter((t) => !suggestedToolNames.has(t.name)).map((t) => t.name);
    const current = new Set(allowedTools);
    onAllowedToolsChange([...allowedTools, ...restNames.filter((n) => !current.has(n))]);
  };

  const handleDisableRest = () => {
    // Disable ALL non-suggested tools (not just the currently filtered subset)
    onAllowedToolsChange(allowedTools.filter((n) => suggestedToolNames.has(n)));
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
          {tools.length > 0 && (
            <Radio.Group
              value={viewMode}
              onChange={(e) => setViewMode(e.target.value)}
              size="small"
              optionType="button"
              buttonStyle="solid"
              options={[
                { label: "Risk Groups", value: "crud" },
                { label: "Flat List", value: "flat" },
              ]}
            />
          )}
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
            <Text className="ml-3">Loading tools from spec...</Text>
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
          keyTools && keyTools.length > 0 ? (
            <div className="text-center py-4 text-gray-400 border rounded-lg border-dashed">
              <ToolOutlined className="text-2xl mb-2" />
              <Text>No tools loaded from spec</Text>
              <Text className="text-sm block mt-1">
                Expected tools: {keyTools.map((t) => t.name).join(", ")}
              </Text>
            </div>
          ) : (
            <div className="text-center py-6 text-gray-400 border rounded-lg border-dashed">
              <ToolOutlined className="text-2xl mb-2" />
              <Text>No tools available for configuration</Text>
              <br />
              <Text className="text-sm">Connect to an MCP server with tools to configure them</Text>
            </div>
          )
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
            <div className="flex items-center gap-2 p-3 bg-green-50 rounded-lg border border-green-200">
              <CheckCircleOutlined className="text-green-600" />
              <Text className="text-green-700 font-medium">
                {allowedTools.length} of {tools.length} {tools.length === 1 ? "tool" : "tools"} enabled for user
                access
              </Text>
            </div>

            {/* Search box shared by both views */}
            <Input
              placeholder="Search tools by name or description..."
              prefix={<SearchOutlined className="text-gray-400" />}
              value={toolSearchTerm}
              onChange={(e) => setToolSearchTerm(e.target.value)}
              allowClear
              className="rounded-lg"
              size="large"
            />

            {/* CRUD grouped view */}
            {viewMode === "crud" && (
              <McpCrudPermissionPanel
                tools={tools}
                searchFilter={toolSearchTerm}
                value={allowedTools.length === 0 ? undefined : allowedTools}
                onChange={(allowed) => onAllowedToolsChange(allowed)}
              />
            )}

            {/* Flat list view */}
            {viewMode === "flat" && (
              <>
                {filteredTools.length === 0 ? (
                  <div className="text-center py-6 text-gray-400 border rounded-lg border-dashed">
                    <SearchOutlined className="text-2xl mb-2" />
                    <Text>No tools found matching &quot;{toolSearchTerm}&quot;</Text>
                  </div>
                ) : (
                  <div className="space-y-2">
                {pinnedFiltered.length > 0 && (
                  <>
                    <div className="flex items-center justify-between px-1">
                      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                        Suggested tools
                      </p>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={handleEnableSuggested}
                          className="text-xs text-blue-600 hover:text-blue-700"
                        >
                          Enable all
                        </button>
                        <button
                          type="button"
                          onClick={handleDisableSuggested}
                          className="text-xs text-gray-500 hover:text-gray-700"
                        >
                          Disable all
                        </button>
                      </div>
                    </div>
                    {pinnedFiltered.map((tool) => (
                      <ToolRow
                        key={tool.name}
                        tool={tool}
                        isEnabled={allowedTools.includes(tool.name)}
                        isEditExpanded={expandedTools.has(tool.name)}
                        toolNameToDisplayName={toolNameToDisplayName}
                        toolNameToDescription={toolNameToDescription}
                        onToggle={handleToolToggle}
                        onToggleExpand={handleToggleEditExpanded}
                        onDisplayNameChange={handleDisplayNameChange}
                        onDescriptionChange={handleDescriptionChange}
                      />
                    ))}
                  </>
                )}
                {restFiltered.length > 0 && (
                  <>
                    <div className="flex items-center justify-between px-1 pt-2">
                      <p className="text-xs font-semibold text-gray-500 uppercase tracking-wide">
                        {pinnedFiltered.length > 0 ? "All tools" : "Tools"}
                      </p>
                      <div className="flex gap-2">
                        <button
                          type="button"
                          onClick={handleEnableRest}
                          className="text-xs text-blue-600 hover:text-blue-700"
                        >
                          Enable all
                        </button>
                        <button
                          type="button"
                          onClick={handleDisableRest}
                          className="text-xs text-gray-500 hover:text-gray-700"
                        >
                          Disable all
                        </button>
                      </div>
                    </div>
                    {restFiltered.map((tool) => (
                      <ToolRow
                        key={tool.name}
                        tool={tool}
                        isEnabled={allowedTools.includes(tool.name)}
                        isEditExpanded={expandedTools.has(tool.name)}
                        toolNameToDisplayName={toolNameToDisplayName}
                        toolNameToDescription={toolNameToDescription}
                        onToggle={handleToolToggle}
                        onToggleExpand={handleToggleEditExpanded}
                        onDisplayNameChange={handleDisplayNameChange}
                        onDescriptionChange={handleDescriptionChange}
                      />
                    ))}
                  </>
                )}
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </Card>
  );
};

export default MCPToolConfiguration;
