import React, { useEffect, useMemo, useRef, useState } from "react";
import { Wrench, CircleCheck, Search, Pencil } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Input } from "@/components/ui/input";
import { InputGroup, InputGroupAddon, InputGroupInput } from "@/components/ui/input-group";
import { Textarea } from "@/components/ui/textarea";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { cn } from "@/lib/cva.config";
import McpCrudPermissionPanel from "@/components/mcp_tools/McpCrudPermissionPanel";
import { TOOL_DISPLAY_NAME_PATTERN } from "./utils";

interface KeyTool {
  name: string;
  description: string;
}

interface MCPToolConfigurationProps {
  accessToken: string | null;
  formValues: Record<string, any>;
  allowedTools: string[];
  existingAllowedTools: string[] | null;
  onAllowedToolsChange: (tools: string[]) => void;
  toolNameToDisplayName: Record<string, string>;
  toolNameToDescription: Record<string, string>;
  onToolNameToDisplayNameChange: (map: Record<string, string>) => void;
  onToolNameToDescriptionChange: (map: Record<string, string>) => void;
  hasToolAllowlistInteraction?: boolean;
  onToolAllowlistInteraction?: () => void;
  /** Curated key tools from the OpenAPI registry preset (shown before spec loads). */
  keyTools?: KeyTool[];
  /** External tool state lifted from parent to avoid duplicate fetch requests. */
  externalTools?: any[];
  externalIsLoading?: boolean;
  externalError?: string | null;
  externalErrorStatus?: number | null;
  externalCanFetch?: boolean;
  /** When true, do not auto-select all tools for servers with no stored allowlist. */
  isEditMode?: boolean;
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
}) => {
  const displayNameValue = toolNameToDisplayName[tool.name] || "";
  const isDisplayNameInvalid = displayNameValue !== "" && !TOOL_DISPLAY_NAME_PATTERN.test(displayNameValue);

  return (
    <div
      className={cn(
        "rounded-lg border transition-colors",
        isEnabled ? "border-primary/40 bg-accent" : "border-border bg-muted",
      )}
    >
      <div className="cursor-pointer p-4" onClick={() => onToggle(tool.name)}>
        <div className="flex items-start gap-3">
          <Checkbox checked={isEnabled} onCheckedChange={() => onToggle(tool.name)} />
          <div className="flex-1">
            <div className="flex items-center gap-2">
              <p className="text-sm font-medium">{toolNameToDisplayName[tool.name] || tool.name}</p>
              <Badge variant={isEnabled ? "secondary" : "outline"}>{isEnabled ? "Enabled" : "Disabled"}</Badge>
              {toolNameToDisplayName[tool.name] && <Badge variant="secondary">Custom name</Badge>}
            </div>
            {(toolNameToDescription[tool.name] || tool.description) && (
              <p className="mt-1 block text-sm text-muted-foreground">
                {toolNameToDescription[tool.name] || tool.description}
              </p>
            )}
            <p className="mt-1 block text-xs text-muted-foreground">
              {isEnabled ? "✓ Users can call this tool" : "✗ Users cannot call this tool"}
            </p>
          </div>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={(e) => onToggleExpand(tool.name, e)}
            title="Edit display name and description"
          >
            <Pencil />
          </Button>
        </div>
      </div>
      {isEditExpanded && (
        <div
          className="space-y-3 rounded-b-lg border-t border-border bg-muted px-4 pt-3 pb-4"
          onClick={(e) => e.stopPropagation()}
        >
          <div>
            <p className="mb-1 block text-xs font-medium">Display Name</p>
            <Input
              placeholder={tool.name}
              value={toolNameToDisplayName[tool.name] || ""}
              onChange={(e) => onDisplayNameChange(tool.name, e.target.value)}
              aria-invalid={isDisplayNameInvalid || undefined}
            />
            {isDisplayNameInvalid ? (
              <p className="mt-1 block text-xs text-destructive">
                Only letters, digits, underscores, and hyphens are allowed (no spaces).
              </p>
            ) : (
              <p className="mt-1 block text-xs text-muted-foreground">
                Override how this tool&apos;s name appears to users. Leave blank to use original.
              </p>
            )}
          </div>
          <div>
            <p className="mb-1 block text-xs font-medium">Description</p>
            <Textarea
              className="field-sizing-fixed"
              placeholder={tool.description || "No description"}
              value={toolNameToDescription[tool.name] || ""}
              onChange={(e) => onDescriptionChange(tool.name, e.target.value)}
              rows={2}
            />
            <p className="mt-1 block text-xs text-muted-foreground">
              Override the tool description shown to users. Leave blank to use original.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

const MCPToolConfiguration: React.FC<MCPToolConfigurationProps> = ({
  accessToken,
  formValues,
  allowedTools,
  existingAllowedTools,
  onAllowedToolsChange,
  toolNameToDisplayName,
  toolNameToDescription,
  onToolNameToDisplayNameChange,
  onToolNameToDescriptionChange,
  hasToolAllowlistInteraction = false,
  onToolAllowlistInteraction,
  keyTools,
  externalTools,
  externalIsLoading,
  externalError,
  externalErrorStatus = null,
  externalCanFetch,
  isEditMode = false,
}) => {
  const previousToolsRef = useRef<ToolEntry[]>([]);
  const [toolSearchTerm, setToolSearchTerm] = useState("");
  const [viewMode, setViewMode] = useState<"crud" | "flat">("crud");
  const hasInitializedRef = useRef(false);
  const previousSuggestedToolNamesRef = useRef<string>("");
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());
  const isPreviewForbidden = externalErrorStatus === 403;

  // Tool list is fetched by the parent (create/edit flow) and passed in. This
  // component renders that state; it never fetches on its own, so there is a
  // single source of truth and no risk of falling back to a different endpoint.
  const tools: ToolEntry[] = externalTools ?? [];
  const isLoadingTools = externalIsLoading ?? false;
  const toolsError = externalError ?? null;
  const canFetchTools = externalCanFetch ?? false;

  // Fuzzy-match curated key tool names against actual loaded tool names
  const suggestedTools = useMemo(() => {
    if (!keyTools || keyTools.length === 0 || tools.length === 0) return [];
    const usedNames = new Set<string>();
    const result: typeof tools = [];
    for (const keyTool of keyTools) {
      const keywords = keyTool.name
        .split("_")
        .map((k) => k.toLowerCase())
        .filter((k) => k.length > 1);
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

  const suggestedToolNames = useMemo(() => new Set(suggestedTools.map((t) => t.name)), [suggestedTools]);

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
    [tools, toolSearchTerm],
  );

  const pinnedFiltered = useMemo(
    () => filteredTools.filter((t) => suggestedToolNames.has(t.name)),
    [filteredTools, suggestedToolNames],
  );

  const restFiltered = useMemo(
    () => filteredTools.filter((t) => !suggestedToolNames.has(t.name)),
    [filteredTools, suggestedToolNames],
  );

  // Auto-select tools when tools are first loaded or when tools list changes
  useEffect(() => {
    const currentToolNames = tools
      .map((tool) => tool.name)
      .sort()
      .join(",");
    const previousToolNames = previousToolsRef.current
      .map((tool) => tool.name)
      .sort()
      .join(",");
    const toolsListChanged = currentToolNames !== previousToolNames;

    // Reset initialization when a new preset is selected (suggestedTools fingerprint changes)
    const currentSuggestedNames = suggestedTools
      .map((t) => t.name)
      .sort()
      .join(",");
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

        if (existingAllowedTools !== null) {
          // Edit mode: honor stored allowlist, including [] (user cleared all tools).
          const validExistingTools = existingAllowedTools.filter((toolName) => availableToolNames.includes(toolName));
          onAllowedToolsChange(validExistingTools);
        } else if (isEditMode) {
          // Unrestricted legacy server: preserve a restored/in-progress
          // selection, otherwise do not auto-select before the user picks tools.
          onAllowedToolsChange(
            hasToolAllowlistInteraction ? allowedTools.filter((toolName) => availableToolNames.includes(toolName)) : [],
          );
        } else if (suggestedTools.length > 0) {
          // OpenAPI preset: only enable suggested tools by default
          onAllowedToolsChange(suggestedTools.map((t) => t.name).filter((name) => availableToolNames.includes(name)));
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
  }, [
    tools,
    allowedTools,
    existingAllowedTools,
    onAllowedToolsChange,
    suggestedTools,
    hasToolAllowlistInteraction,
    isEditMode,
  ]);

  const isLegacyUnrestrictedEdit =
    isEditMode && existingAllowedTools === null && allowedTools.length === 0 && !hasToolAllowlistInteraction;
  const effectiveAllowedTools = useMemo(
    () => (isLegacyUnrestrictedEdit ? tools.map((tool) => tool.name) : allowedTools),
    [allowedTools, isLegacyUnrestrictedEdit, tools],
  );
  const effectiveAllowedToolNames = useMemo(() => new Set(effectiveAllowedTools), [effectiveAllowedTools]);

  const handleAllowedToolsChange = (nextAllowedTools: string[]) => {
    onToolAllowlistInteraction?.();
    onAllowedToolsChange(nextAllowedTools);
  };

  const handleToolToggle = (toolName: string) => {
    if (effectiveAllowedToolNames.has(toolName)) {
      handleAllowedToolsChange(effectiveAllowedTools.filter((name) => name !== toolName));
    } else {
      handleAllowedToolsChange([...effectiveAllowedTools, toolName]);
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
    const missingSuggestedNames = suggestedNames.filter((name) => !effectiveAllowedToolNames.has(name));
    if (missingSuggestedNames.length === 0) return;
    handleAllowedToolsChange([...effectiveAllowedTools, ...missingSuggestedNames]);
  };

  const handleDisableSuggested = () => {
    // Disable ALL suggested tools (not just the currently filtered subset)
    handleAllowedToolsChange(effectiveAllowedTools.filter((n) => !suggestedToolNames.has(n)));
  };

  const handleEnableRest = () => {
    // Enable ALL non-suggested tools (not just the currently filtered subset)
    const restNames = tools.filter((t) => !suggestedToolNames.has(t.name)).map((t) => t.name);
    const missingRestNames = restNames.filter((n) => !effectiveAllowedToolNames.has(n));
    if (missingRestNames.length === 0) return;
    handleAllowedToolsChange([...effectiveAllowedTools, ...missingRestNames]);
  };

  const handleDisableRest = () => {
    // Disable ALL non-suggested tools (not just the currently filtered subset)
    handleAllowedToolsChange(effectiveAllowedTools.filter((n) => suggestedToolNames.has(n)));
  };

  // Don't show anything if required fields aren't filled
  if (!canFetchTools && !formValues.url && !formValues.spec_path) {
    return null;
  }

  return (
    <Card className="p-6">
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Wrench className="size-4 text-muted-foreground" />
            <h3 className="text-lg font-medium">Tool Configuration</h3>
            {tools.length > 0 && <Badge variant="secondary">{tools.length}</Badge>}
          </div>
          {tools.length > 0 && (
            <div className="flex items-center gap-1">
              <Button
                size="sm"
                variant={viewMode === "crud" ? "default" : "outline"}
                onClick={() => setViewMode("crud")}
              >
                Risk Groups
              </Button>
              <Button
                size="sm"
                variant={viewMode === "flat" ? "default" : "outline"}
                onClick={() => setViewMode("flat")}
              >
                Flat List
              </Button>
            </div>
          )}
        </div>

        {/* Description */}
        <div className="rounded-lg border border-border bg-muted p-3">
          <p className="text-sm">
            <strong>Select which tools users can call:</strong> Only checked tools will be available for users to
            invoke. Unchecked tools will be blocked from execution.
          </p>
        </div>

        {/* Loading state */}
        {isLoadingTools && (
          <div className="flex items-center justify-center gap-3 py-6">
            <UiLoadingSpinner className="size-6 text-muted-foreground" />
            <p className="text-sm">Loading tools from spec...</p>
          </div>
        )}

        {/* Error state */}
        {toolsError && !isLoadingTools && isPreviewForbidden && (
          <div className="rounded-lg border border-border bg-muted p-4">
            <p className="text-sm">{toolsError}</p>
          </div>
        )}

        {toolsError && !isLoadingTools && !isPreviewForbidden && (
          <div className="rounded-lg border border-dashed border-destructive/40 bg-destructive/5 py-6 text-center">
            <Wrench className="mx-auto mb-2 size-6 text-destructive" />
            <p className="text-sm font-medium text-destructive">Unable to load tools</p>
            <p className="text-sm text-destructive">{toolsError}</p>
          </div>
        )}

        {/* No tools state */}
        {!isLoadingTools &&
          !toolsError &&
          tools.length === 0 &&
          canFetchTools &&
          (keyTools && keyTools.length > 0 ? (
            <div className="rounded-lg border border-dashed py-4 text-center text-muted-foreground">
              <Wrench className="mx-auto mb-2 size-6" />
              <p className="text-sm">No tools loaded from spec</p>
              <p className="mt-1 block text-sm">Expected tools: {keyTools.map((t) => t.name).join(", ")}</p>
            </div>
          ) : (
            <div className="rounded-lg border border-dashed py-6 text-center text-muted-foreground">
              <Wrench className="mx-auto mb-2 size-6" />
              <p className="text-sm">No tools available for configuration</p>
              <p className="text-sm">Connect to an MCP server with tools to configure them</p>
            </div>
          ))}

        {/* Incomplete form state */}
        {!canFetchTools && (formValues.url || formValues.spec_path) && (
          <div className="rounded-lg border border-dashed py-6 text-center text-muted-foreground">
            <Wrench className="mx-auto mb-2 size-6" />
            <p className="text-sm">Complete required fields to configure tools</p>
            <p className="text-sm">Fill in URL, Transport, and Authentication to load available tools</p>
          </div>
        )}

        {/* Tools loaded successfully */}
        {!isLoadingTools && !toolsError && tools.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center gap-2 rounded-lg border border-border bg-muted p-3">
              <CircleCheck className="size-4" />
              <p className="text-sm font-medium">
                {effectiveAllowedTools.length} of {tools.length} {tools.length === 1 ? "tool" : "tools"} enabled for
                user access
              </p>
            </div>

            {/* Search box shared by both views */}
            <InputGroup className="w-full">
              <InputGroupAddon>
                <Search className="size-4 text-muted-foreground" />
              </InputGroupAddon>
              <InputGroupInput
                placeholder="Search tools by name or description..."
                value={toolSearchTerm}
                onChange={(e) => setToolSearchTerm(e.target.value)}
              />
            </InputGroup>

            {/* CRUD grouped view */}
            {viewMode === "crud" && (
              <McpCrudPermissionPanel
                tools={tools}
                searchFilter={toolSearchTerm}
                value={isLegacyUnrestrictedEdit ? undefined : allowedTools}
                onChange={handleAllowedToolsChange}
              />
            )}

            {/* Flat list view */}
            {viewMode === "flat" && (
              <>
                {filteredTools.length === 0 ? (
                  <div className="rounded-lg border border-dashed py-6 text-center text-muted-foreground">
                    <Search className="mx-auto mb-2 size-6" />
                    <p className="text-sm">No tools found matching &quot;{toolSearchTerm}&quot;</p>
                  </div>
                ) : (
                  <div className="space-y-2">
                    {pinnedFiltered.length > 0 && (
                      <>
                        <div className="flex items-center justify-between px-1">
                          <p className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                            Suggested tools
                          </p>
                          <div className="flex gap-2">
                            <Button variant="link" size="sm" onClick={handleEnableSuggested}>
                              Enable all
                            </Button>
                            <Button variant="link" size="sm" onClick={handleDisableSuggested}>
                              Disable all
                            </Button>
                          </div>
                        </div>
                        {pinnedFiltered.map((tool) => (
                          <ToolRow
                            key={tool.name}
                            tool={tool}
                            isEnabled={effectiveAllowedToolNames.has(tool.name)}
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
                          <p className="text-xs font-semibold tracking-wide text-muted-foreground uppercase">
                            {pinnedFiltered.length > 0 ? "All tools" : "Tools"}
                          </p>
                          <div className="flex gap-2">
                            <Button variant="link" size="sm" onClick={handleEnableRest}>
                              Enable all
                            </Button>
                            <Button variant="link" size="sm" onClick={handleDisableRest}>
                              Disable all
                            </Button>
                          </div>
                        </div>
                        {restFiltered.map((tool) => (
                          <ToolRow
                            key={tool.name}
                            tool={tool}
                            isEnabled={effectiveAllowedToolNames.has(tool.name)}
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
