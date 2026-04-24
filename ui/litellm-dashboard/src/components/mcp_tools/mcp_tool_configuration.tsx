import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Wrench as ToolOutlined,
  CheckCircle2 as CheckCircleOutlined,
  Search as SearchOutlined,
  Pencil as EditOutlined,
  LoaderCircle,
} from "lucide-react";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
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
        ? "bg-primary/5 border-primary/40 hover:border-primary/60"
        : "bg-muted border-border hover:border-muted-foreground/40"
    }`}
  >
    <div className="p-4 cursor-pointer" onClick={() => onToggle(tool.name)}>
      <div className="flex items-start gap-3">
        <Checkbox
          checked={isEnabled}
          onCheckedChange={() => onToggle(tool.name)}
          onClick={(e) => e.stopPropagation()}
        />
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <span className="font-medium text-foreground">
              {toolNameToDisplayName[tool.name] || tool.name}
            </span>
            <Badge variant={isEnabled ? "secondary" : "outline"} className="text-xs">
              {isEnabled ? "Enabled" : "Disabled"}
            </Badge>
            {toolNameToDisplayName[tool.name] && (
              <Badge variant="outline" className="text-xs">
                Custom name
              </Badge>
            )}
          </div>
          {(toolNameToDescription[tool.name] || tool.description) && (
            <span className="text-muted-foreground text-sm block mt-1">
              {toolNameToDescription[tool.name] || tool.description}
            </span>
          )}
          <span className="text-muted-foreground text-xs block mt-1">
            {isEnabled ? "✓ Users can call this tool" : "✗ Users cannot call this tool"}
          </span>
        </div>
        <button
          type="button"
          onClick={(e) => onToggleExpand(tool.name, e)}
          className={`p-1.5 rounded-md transition-colors ${
            isEditExpanded
              ? "bg-primary/10 text-primary"
              : "text-muted-foreground hover:text-foreground hover:bg-muted"
          }`}
          title="Edit display name and description"
        >
          <EditOutlined className="h-4 w-4" />
        </button>
      </div>
    </div>
    {isEditExpanded && (
      <div
        className="px-4 pb-4 pt-3 border-t border-border space-y-3 bg-muted rounded-b-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <div>
          <span className="text-xs font-medium text-muted-foreground mb-1 block">
            Display Name
          </span>
          <Input
            placeholder={tool.name}
            value={toolNameToDisplayName[tool.name] || ""}
            onChange={(e) => onDisplayNameChange(tool.name, e.target.value)}
          />
          <span className="text-xs text-muted-foreground mt-1 block">
            Override how this tool&apos;s name appears to users. Leave blank to use original.
          </span>
        </div>
        <div>
          <span className="text-xs font-medium text-muted-foreground mb-1 block">Description</span>
          <Textarea
            placeholder={tool.description || "No description"}
            value={toolNameToDescription[tool.name] || ""}
            onChange={(e) => onDescriptionChange(tool.name, e.target.value)}
            rows={2}
          />
          <span className="text-xs text-muted-foreground mt-1 block">
            Override the tool description shown to users. Leave blank to use original.
          </span>
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

  useEffect(() => {
    const currentToolNames = tools.map((tool) => tool.name).sort().join(",");
    const previousToolNames = previousToolsRef.current.map((tool) => tool.name).sort().join(",");
    const toolsListChanged = currentToolNames !== previousToolNames;

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
          const validExistingTools = existingAllowedTools.filter((toolName) =>
            availableToolNames.includes(toolName)
          );
          onAllowedToolsChange(validExistingTools);
        } else if (suggestedTools.length > 0) {
          onAllowedToolsChange(
            suggestedTools.map((t) => t.name).filter((name) => availableToolNames.includes(name))
          );
        } else {
          onAllowedToolsChange(availableToolNames);
        }
      } else {
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
    const suggestedNames = suggestedTools.map((t) => t.name);
    const others = allowedTools.filter((n) => !suggestedToolNames.has(n));
    onAllowedToolsChange([...others, ...suggestedNames]);
  };

  const handleDisableSuggested = () => {
    onAllowedToolsChange(allowedTools.filter((n) => !suggestedToolNames.has(n)));
  };

  const handleEnableRest = () => {
    const restNames = tools.filter((t) => !suggestedToolNames.has(t.name)).map((t) => t.name);
    const current = new Set(allowedTools);
    onAllowedToolsChange([...allowedTools, ...restNames.filter((n) => !current.has(n))]);
  };

  const handleDisableRest = () => {
    onAllowedToolsChange(allowedTools.filter((n) => suggestedToolNames.has(n)));
  };

  if (!canFetchTools && !formValues.url && !formValues.spec_path) {
    return null;
  }

  return (
    <Card className="p-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <ToolOutlined className="text-primary h-5 w-5" />
            <h3 className="text-lg font-semibold">Tool Configuration</h3>
            {tools.length > 0 && (
              <Badge variant="secondary">{tools.length}</Badge>
            )}
          </div>
          {tools.length > 0 && (
            <ToggleGroup
              type="single"
              value={viewMode}
              onValueChange={(v) => {
                if (v) setViewMode(v as "crud" | "flat");
              }}
              size="sm"
            >
              <ToggleGroupItem value="crud">Risk Groups</ToggleGroupItem>
              <ToggleGroupItem value="flat">Flat List</ToggleGroupItem>
            </ToggleGroup>
          )}
        </div>

        <div className="bg-primary/5 border border-primary/20 rounded-lg p-3">
          <span className="text-foreground text-sm">
            <strong>Select which tools users can call:</strong> Only checked tools will be available for users to
            invoke. Unchecked tools will be blocked from execution.
          </span>
        </div>

        {isLoadingTools && (
          <div className="flex items-center justify-center py-6">
            <LoaderCircle className="h-6 w-6 animate-spin text-primary" />
            <span className="ml-3">Loading tools from spec...</span>
          </div>
        )}

        {toolsError && !isLoadingTools && (
          <div className="text-center py-6 text-destructive border rounded-lg border-dashed border-destructive/50 bg-destructive/5">
            <ToolOutlined className="text-2xl mb-2 mx-auto h-6 w-6" />
            <span className="text-destructive font-medium block">Unable to load tools</span>
            <span className="text-sm text-destructive block">{toolsError}</span>
          </div>
        )}

        {!isLoadingTools && !toolsError && tools.length === 0 && canFetchTools && (
          keyTools && keyTools.length > 0 ? (
            <div className="text-center py-4 text-muted-foreground border rounded-lg border-dashed">
              <ToolOutlined className="text-2xl mb-2 mx-auto h-6 w-6" />
              <span>No tools loaded from spec</span>
              <span className="text-sm block mt-1">
                Expected tools: {keyTools.map((t) => t.name).join(", ")}
              </span>
            </div>
          ) : (
            <div className="text-center py-6 text-muted-foreground border rounded-lg border-dashed">
              <ToolOutlined className="text-2xl mb-2 mx-auto h-6 w-6" />
              <span className="block">No tools available for configuration</span>
              <span className="text-sm">Connect to an MCP server with tools to configure them</span>
            </div>
          )
        )}

        {!canFetchTools && (formValues.url || formValues.spec_path) && (
          <div className="text-center py-6 text-muted-foreground border rounded-lg border-dashed">
            <ToolOutlined className="text-2xl mb-2 mx-auto h-6 w-6" />
            <span className="block">Complete required fields to configure tools</span>
            <span className="text-sm">Fill in URL, Transport, and Authentication to load available tools</span>
          </div>
        )}

        {!isLoadingTools && !toolsError && tools.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2 p-3 bg-primary/5 rounded-lg border border-primary/20">
              <CheckCircleOutlined className="text-primary h-5 w-5" />
              <span className="text-foreground font-medium">
                {allowedTools.length} of {tools.length} {tools.length === 1 ? "tool" : "tools"} enabled for user
                access
              </span>
            </div>

            <div className="relative">
              <SearchOutlined className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search tools by name or description..."
                value={toolSearchTerm}
                onChange={(e) => setToolSearchTerm(e.target.value)}
                className="pl-9"
              />
            </div>

            {viewMode === "crud" && (
              <McpCrudPermissionPanel
                tools={tools}
                searchFilter={toolSearchTerm}
                value={allowedTools.length === 0 ? undefined : allowedTools}
                onChange={(allowed) => onAllowedToolsChange(allowed)}
              />
            )}

            {viewMode === "flat" && (
              <>
                {filteredTools.length === 0 ? (
                  <div className="text-center py-6 text-muted-foreground border rounded-lg border-dashed">
                    <SearchOutlined className="text-2xl mb-2 mx-auto h-6 w-6" />
                    <span>No tools found matching &quot;{toolSearchTerm}&quot;</span>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {pinnedFiltered.length > 0 && (
                      <>
                        <div className="flex items-center justify-between px-1">
                          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                            Suggested tools
                          </p>
                          <div className="flex gap-2">
                            <button
                              type="button"
                              onClick={handleEnableSuggested}
                              className="text-xs text-primary hover:text-primary/80"
                            >
                              Enable all
                            </button>
                            <button
                              type="button"
                              onClick={handleDisableSuggested}
                              className="text-xs text-muted-foreground hover:text-foreground"
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
                          <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wide">
                            {pinnedFiltered.length > 0 ? "All tools" : "Tools"}
                          </p>
                          <div className="flex gap-2">
                            <button
                              type="button"
                              onClick={handleEnableRest}
                              className="text-xs text-primary hover:text-primary/80"
                            >
                              Enable all
                            </button>
                            <button
                              type="button"
                              onClick={handleDisableRest}
                              className="text-xs text-muted-foreground hover:text-foreground"
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
