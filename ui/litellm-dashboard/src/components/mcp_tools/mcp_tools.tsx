import React, { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { ToolTestPanel } from "./ToolTestPanel";
import { MCPTool, MCPToolsViewerProps, MCPContent, CallMCPToolResponse } from "./types";
import { listMCPTools, callMCPTool } from "../networking";

import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import {
  Bot as RobotOutlined,
  Wrench as ToolOutlined,
  Search as SearchOutlined,
  Key as KeyOutlined,
  Check as CheckIcon,
} from "lucide-react";

const MCPToolsViewer = ({
  serverId,
  accessToken,
  auth_type,
  userRole,
  userID,
  serverAlias,
  extraHeaders,
}: MCPToolsViewerProps) => {
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null);
  const [toolResult, setToolResult] = useState<MCPContent[] | null>(null);
  const [toolError, setToolError] = useState<Error | null>(null);
  const [toolSearchTerm, setToolSearchTerm] = useState("");

  const [passthroughHeaders, setPassthroughHeaders] = useState<Record<string, string>>({});
  const [showHeaderInput, setShowHeaderInput] = useState(false);

  const hasExtraHeaders = extraHeaders && extraHeaders.length > 0;

  const buildCustomHeaders = () => {
    if (!serverAlias || !hasExtraHeaders) return undefined;

    const customHeaders: Record<string, string> = {};

    Object.entries(passthroughHeaders).forEach(([headerName, headerValue]) => {
      if (headerValue && headerValue.trim()) {
        const mcpHeaderName = `x-mcp-${serverAlias}-${headerName.toLowerCase()}`;
        customHeaders[mcpHeaderName] = headerValue;
      }
    });

    return Object.keys(customHeaders).length > 0 ? customHeaders : undefined;
  };

  const {
    data: mcpToolsResponse,
    isLoading: isLoadingTools,
    error: mcpToolsError,
    refetch: refetchTools,
  } = useQuery({
    queryKey: ["mcpTools", serverId, passthroughHeaders],
    queryFn: () => {
      if (!accessToken) throw new Error("Access Token required");
      return listMCPTools(accessToken, serverId, buildCustomHeaders());
    },
    enabled: !!accessToken,
    staleTime: 30000,
  });

  const { mutate: executeTool, isPending: isCallingTool } = useMutation({
    mutationFn: async (args: { tool: MCPTool; arguments: Record<string, any> }) => {
      if (!accessToken) throw new Error("Access Token required");

      try {
        const result: CallMCPToolResponse = await callMCPTool(
          accessToken,
          serverId,
          args.tool.name,
          args.arguments,
          { customHeaders: buildCustomHeaders() },
        );
        return result;
      } catch (error) {
        throw error;
      }
    },
    onSuccess: (data) => {
      setToolResult(data.content);
      setToolError(null);
    },
    onError: (error: Error) => {
      setToolError(error);
      setToolResult(null);
    },
  });

  const toolsData = mcpToolsResponse?.tools || [];

  const filteredTools = toolsData.filter((tool: MCPTool) => {
    const searchLower = toolSearchTerm.toLowerCase();
    return (
      tool.name.toLowerCase().includes(searchLower) ||
      (tool.description && tool.description.toLowerCase().includes(searchLower)) ||
      (tool.mcp_info.server_name && tool.mcp_info.server_name.toLowerCase().includes(searchLower))
    );
  });

  return (
    <div className="w-full h-screen p-4 bg-background">
      <Card className="w-full rounded-xl shadow-md overflow-hidden">
        <div className="flex h-auto w-full gap-4">
          {/* Left Sidebar with Controls */}
          <div className="w-1/4 p-4 bg-muted flex flex-col">
            <h2 className="text-xl font-semibold mb-6 mt-2 text-foreground">MCP Tools</h2>

            <div className="flex flex-col flex-1">
              {/* Extra Headers Input Section */}
              {hasExtraHeaders && (
                <div className="mb-4 p-3 bg-primary/5 border border-primary/20 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center">
                      <KeyOutlined className="text-primary mr-2 h-4 w-4" />
                      <span className="text-sm font-medium text-foreground">
                        Additional Headers
                      </span>
                    </div>
                    <Button
                      size="sm"
                      variant="link"
                      onClick={() => setShowHeaderInput(!showHeaderInput)}
                      className="text-primary p-0 h-auto"
                    >
                      {showHeaderInput ? "Hide" : "Configure"}
                    </Button>
                  </div>

                  {!showHeaderInput && Object.keys(passthroughHeaders).length === 0 && (
                    <span className="text-xs text-muted-foreground">
                      This server requires additional headers. Click &quot;Configure&quot; to provide values.
                    </span>
                  )}

                  {showHeaderInput && (
                    <div className="mt-3 space-y-2">
                      {extraHeaders?.map((headerName) => (
                        <div key={headerName}>
                          <label className="block text-xs font-medium text-foreground mb-1">
                            {headerName}
                          </label>
                          <div className="relative">
                            <KeyOutlined className="absolute left-3 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-muted-foreground" />
                            <Input
                              placeholder={`Enter ${headerName}`}
                              value={passthroughHeaders[headerName] || ""}
                              onChange={(e) => {
                                setPassthroughHeaders({
                                  ...passthroughHeaders,
                                  [headerName]: e.target.value,
                                });
                              }}
                              className="pl-9 h-8"
                            />
                          </div>
                        </div>
                      ))}
                      <Button
                        size="sm"
                        onClick={() => {
                          refetchTools();
                          setShowHeaderInput(false);
                        }}
                        disabled={Object.values(passthroughHeaders).every((v) => !v || !v.trim())}
                        className="w-full mt-2"
                      >
                        Load Tools
                      </Button>
                    </div>
                  )}

                  {!showHeaderInput && Object.keys(passthroughHeaders).length > 0 && (
                    <div className="mt-2">
                      <span className="text-xs text-green-700 dark:text-green-400 flex items-center">
                        <span className="inline-block w-2 h-2 bg-green-500 rounded-full mr-2"></span>
                        {Object.keys(passthroughHeaders).length} header(s) configured
                      </span>
                    </div>
                  )}
                </div>
              )}

              {/* Tool Selection - Show tools first */}
              <div className="flex flex-col flex-1 min-h-0">
                <span className="font-medium block mb-3 text-foreground flex items-center">
                  <ToolOutlined className="mr-2 h-4 w-4" /> Available Tools
                  {toolsData.length > 0 && (
                    <span className="ml-2 bg-primary/10 text-primary text-xs font-medium px-2 py-0.5 rounded-full">
                      {toolsData.length}
                    </span>
                  )}
                </span>

                {toolsData.length > 0 && (
                  <div className="mb-3 relative">
                    <SearchOutlined className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                    <Input
                      placeholder="Search tools..."
                      value={toolSearchTerm}
                      onChange={(e) => setToolSearchTerm(e.target.value)}
                      className="pl-9"
                    />
                  </div>
                )}

                {isLoadingTools && (
                  <div className="flex flex-col items-center justify-center py-8 bg-background border border-border rounded-lg">
                    <div className="relative mb-3">
                      <div className="animate-spin rounded-full h-6 w-6 border-2 border-muted"></div>
                      <div className="animate-spin rounded-full h-6 w-6 border-2 border-primary border-t-transparent absolute top-0"></div>
                    </div>
                    <p className="text-xs font-medium text-foreground">Loading tools...</p>
                  </div>
                )}

                {mcpToolsResponse?.error && !isLoadingTools && !toolsData.length && (
                  <div className="p-3 text-xs text-destructive rounded-lg bg-destructive/10 border border-destructive/30">
                    <p className="font-medium">Error: {mcpToolsResponse.message}</p>
                  </div>
                )}

                {!isLoadingTools && !mcpToolsResponse?.error && (!toolsData || toolsData.length === 0) && (
                  <div className="p-4 text-center bg-background border border-border rounded-lg">
                    <div className="mx-auto w-8 h-8 bg-muted rounded-full flex items-center justify-center mb-2">
                      <ToolOutlined className="w-4 h-4 text-muted-foreground" />
                    </div>
                    <p className="text-xs font-medium text-foreground mb-1">No tools available</p>
                    <p className="text-xs text-muted-foreground">No tools found for this server</p>
                  </div>
                )}

                {!isLoadingTools && !mcpToolsResponse?.error && toolsData.length > 0 && (
                  <>
                    {filteredTools.length === 0 ? (
                      <div className="p-4 text-center bg-background border border-border rounded-lg">
                        <SearchOutlined className="text-2xl text-muted-foreground mb-2 mx-auto h-5 w-5" />
                        <p className="text-xs font-medium text-foreground mb-1">No tools found</p>
                        <p className="text-xs text-muted-foreground">No tools match &quot;{toolSearchTerm}&quot;</p>
                      </div>
                    ) : (
                      <div
                        className="space-y-2 flex-1 overflow-y-auto min-h-0 mcp-tools-scrollable"
                        style={{
                          maxHeight: "400px",
                          scrollbarWidth: "auto",
                        }}
                      >
                        {filteredTools.map((tool: MCPTool) => (
                          <div
                            key={tool.name}
                            className={`border rounded-lg p-3 cursor-pointer transition-all hover:shadow-sm ${
                              selectedTool?.name === tool.name
                                ? "border-primary bg-primary/5 ring-1 ring-primary/40"
                                : "border-border bg-background hover:border-muted-foreground/40"
                            }`}
                            onClick={() => {
                              setSelectedTool(tool);
                              setToolResult(null);
                              setToolError(null);
                            }}
                          >
                            <div className="flex items-start space-x-2">
                              {tool.mcp_info.logo_url && (
                                <img
                                  src={tool.mcp_info.logo_url}
                                  alt={`${tool.mcp_info.server_name} logo`}
                                  className="w-4 h-4 object-contain flex-shrink-0 mt-0.5"
                                />
                              )}
                              <div className="flex-1 min-w-0">
                                <h4 className="font-mono text-xs font-medium text-foreground truncate">
                                  {tool.name}
                                </h4>
                                <p className="text-xs text-muted-foreground truncate">
                                  {tool.mcp_info.server_name}
                                </p>
                                <p className="text-xs text-muted-foreground mt-1 line-clamp-2 leading-relaxed">
                                  {tool.description}
                                </p>
                              </div>
                            </div>
                            {selectedTool?.name === tool.name && (
                              <div className="mt-2 pt-2 border-t border-primary/30">
                                <div className="flex items-center text-xs font-medium text-primary">
                                  <CheckIcon className="w-3 h-3 mr-1" />
                                  Selected
                                </div>
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
          </div>

          {/* Main Testing Area */}
          <div className="w-3/4 flex flex-col bg-background">
            <div className="p-4 border-b border-border flex justify-between items-center">
              <h2 className="text-xl font-semibold mb-0 text-foreground">Tool Testing Playground</h2>
            </div>

            <div className="flex-1 overflow-auto p-4">
              {!selectedTool ? (
                <div className="h-full flex flex-col items-center justify-center text-muted-foreground">
                  <RobotOutlined className="h-12 w-12 mb-4" />
                  <span className="text-lg font-medium text-foreground mb-2">
                    Select a Tool to Test
                  </span>
                  <span className="text-center text-muted-foreground max-w-md">
                    Choose a tool from the left sidebar to start testing its functionality with custom
                    inputs.
                  </span>
                </div>
              ) : (
                <div className="h-full">
                  <ToolTestPanel
                    tool={selectedTool}
                    onSubmit={(args) => {
                      executeTool({ tool: selectedTool, arguments: args });
                    }}
                    result={toolResult}
                    error={toolError}
                    isLoading={isCallingTool}
                    onClose={() => setSelectedTool(null)}
                  />
                </div>
              )}
            </div>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default MCPToolsViewer;
