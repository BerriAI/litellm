import React, { useState, useEffect } from "react";
import { ServerIcon, ChevronDownIcon, ChevronRightIcon } from "@heroicons/react/outline";
import { Badge } from "@/components/ui/badge";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { fetchMCPServers, fetchMCPToolsets } from "../networking";
import { MCPServer, MCPToolset } from "../mcp_tools/types";
import { ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL } from "../mcp_tools/constants";

interface MCPServerPermissionsProps {
  mcpServers: string[];
  mcpAccessGroups?: string[];
  mcpToolPermissions?: Record<string, string[]>;
  mcpToolsets?: string[];
  accessToken?: string | null;
}

export function MCPServerPermissions({
  mcpServers,
  mcpAccessGroups = [],
  mcpToolPermissions = {},
  mcpToolsets = [],
  accessToken,
}: MCPServerPermissionsProps) {
  const [mcpServerDetails, setMCPServerDetails] = useState<MCPServer[]>([]);
  const [toolsetDetails, setToolsetDetails] = useState<MCPToolset[]>([]);
  const [expandedServers, setExpandedServers] = useState<Set<string>>(new Set());
  const [expandedToolsets, setExpandedToolsets] = useState<Set<string>>(new Set());

  const toggleServerExpansion = (serverId: string) => {
    setExpandedServers((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(serverId)) {
        newSet.delete(serverId);
      } else {
        newSet.add(serverId);
      }
      return newSet;
    });
  };

  const toggleToolsetExpansion = (toolsetId: string) => {
    setExpandedToolsets((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(toolsetId)) {
        newSet.delete(toolsetId);
      } else {
        newSet.add(toolsetId);
      }
      return newSet;
    });
  };

  // Fetch MCP server details when component mounts
  useEffect(() => {
    const fetchMCPServerDetails = async () => {
      if (accessToken && mcpServers.length > 0) {
        try {
          const response = await fetchMCPServers(accessToken);
          if (response && Array.isArray(response)) {
            setMCPServerDetails(response);
          } else if (response.data && Array.isArray(response.data)) {
            setMCPServerDetails(response.data);
          }
        } catch (error) {
          console.error("Error fetching MCP servers:", error);
        }
      }
    };
    fetchMCPServerDetails();
  }, [accessToken, mcpServers.length]);

  // Fetch toolset details
  useEffect(() => {
    const fetchToolsets = async () => {
      if (accessToken && mcpToolsets.length > 0) {
        try {
          const all = await fetchMCPToolsets(accessToken);
          const filtered = Array.isArray(all) ? all.filter((t: MCPToolset) => mcpToolsets.includes(t.toolset_id)) : [];
          setToolsetDetails(filtered);
        } catch (error) {
          console.error("Error fetching toolsets:", error);
        }
      }
    };
    fetchToolsets();
  }, [accessToken, mcpToolsets.length]);

  // Function to get display name for MCP server
  const getMCPServerDisplayName = (serverId: string) => {
    const serverDetail = mcpServerDetails.find((server) => server.server_id === serverId);
    if (serverDetail) {
      const truncatedId = serverId.length > 7 ? `${serverId.slice(0, 3)}...${serverId.slice(-4)}` : serverId;
      return `${serverDetail.alias || serverDetail.server_name || serverId} (${truncatedId})`;
    }
    return serverId;
  };

  const blocksAllMcpServers = mcpServers.includes(NO_MCP_SERVERS_SENTINEL);
  const grantsAllProxyMcpServers = mcpServers.includes(ALL_PROXY_MCP_SERVERS_SENTINEL);

  // Merge servers and access groups into one list
  const mergedItems = [
    ...mcpServers
      .filter((server) => server !== NO_MCP_SERVERS_SENTINEL && server !== ALL_PROXY_MCP_SERVERS_SENTINEL)
      .map((server) => ({ type: "server", value: server })),
    ...mcpAccessGroups.map((group) => ({ type: "accessGroup", value: group })),
  ];
  const totalCount = mergedItems.length + mcpToolsets.length;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <ServerIcon className="h-4 w-4 text-info" />
        <p className="text-sm font-semibold text-foreground">MCP Servers</p>
        <Badge variant={blocksAllMcpServers ? "destructive" : "secondary"}>
          {blocksAllMcpServers ? "Blocked" : grantsAllProxyMcpServers ? "All" : totalCount}
        </Badge>
      </div>

      {blocksAllMcpServers ? (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-destructive/10 border border-destructive/20">
          <ServerIcon className="h-4 w-4 text-destructive" />
          <p className="text-destructive text-sm">
            No MCP servers — this key is blocked from all MCP servers, including its team&apos;s servers
          </p>
        </div>
      ) : grantsAllProxyMcpServers ? (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-info/10 border border-info/20">
          <ServerIcon className="h-4 w-4 text-info" />
          <p className="text-info text-sm">All Proxy MCP Servers</p>
        </div>
      ) : totalCount > 0 ? (
        <div className="max-h-[400px] overflow-y-auto space-y-2 pr-1">
          {mergedItems.map((item, index) => {
            const toolsForServer = item.type === "server" ? mcpToolPermissions[item.value] : undefined;
            const hasToolRestrictions = toolsForServer && toolsForServer.length > 0;
            const isExpanded = expandedServers.has(item.value);

            return (
              <div key={index} className="space-y-2">
                <div
                  onClick={() => hasToolRestrictions && toggleServerExpansion(item.value)}
                  className={`flex items-center gap-3 py-2 px-3 rounded-lg border border-border transition-all ${
                    hasToolRestrictions ? "cursor-pointer hover:bg-accent" : "bg-card"
                  }`}
                >
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    {item.type === "server" ? (
                      <Tooltip>
                        <TooltipTrigger render={<div className="inline-flex items-center gap-2 min-w-0" />}>
                          <span className="inline-block w-1.5 h-1.5 bg-info rounded-full shrink-0"></span>
                          <span className="text-sm font-medium text-foreground truncate">
                            {getMCPServerDisplayName(item.value)}
                          </span>
                        </TooltipTrigger>
                        <TooltipContent>{`Full ID: ${item.value}`}</TooltipContent>
                      </Tooltip>
                    ) : (
                      <div className="inline-flex items-center gap-2 min-w-0">
                        <span className="inline-block w-1.5 h-1.5 bg-success rounded-full shrink-0"></span>
                        <span className="text-sm font-medium text-foreground truncate">{item.value}</span>
                        <span className="ml-1 px-1.5 py-0.5 text-[9px] font-semibold text-success bg-success/10 border border-success/20 rounded-sm uppercase tracking-wide shrink-0">
                          Group
                        </span>
                      </div>
                    )}
                  </div>

                  {hasToolRestrictions && (
                    <div className="flex items-center gap-1 shrink-0 whitespace-nowrap">
                      <span className="text-xs font-medium text-muted-foreground">{toolsForServer.length}</span>
                      <span className="text-xs text-muted-foreground">
                        {toolsForServer.length === 1 ? "tool" : "tools"}
                      </span>
                      {isExpanded ? (
                        <ChevronDownIcon className="h-3.5 w-3.5 text-muted-foreground ml-0.5" />
                      ) : (
                        <ChevronRightIcon className="h-3.5 w-3.5 text-muted-foreground ml-0.5" />
                      )}
                    </div>
                  )}
                </div>

                {/* Show tool permissions if expanded */}
                {hasToolRestrictions && isExpanded && (
                  <div className="ml-4 pl-4 border-l-2 border-info/20 pb-1">
                    <div className="flex flex-wrap gap-1.5">
                      {toolsForServer.map((tool, toolIndex) => (
                        <span
                          key={toolIndex}
                          className="inline-flex items-center px-2.5 py-1 rounded-lg bg-info/10 border border-info/20 text-info text-xs font-medium"
                        >
                          {tool}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            );
          })}

          {/* Toolsets section */}
          {mcpToolsets.length > 0 &&
            mcpToolsets.map((toolsetId, index) => {
              const detail = toolsetDetails.find((t) => t.toolset_id === toolsetId);
              const isExpanded = expandedToolsets.has(toolsetId);
              const toolCount = detail?.tools.length ?? 0;

              return (
                <div key={`toolset-${index}`} className="space-y-2">
                  <div
                    onClick={() => toolCount > 0 && toggleToolsetExpansion(toolsetId)}
                    className={`flex items-center gap-3 py-2 px-3 rounded-lg border border-purple-200 transition-all ${
                      toolCount > 0
                        ? "cursor-pointer hover:bg-purple-50 hover:border-purple-300 dark:hover:bg-purple-950 dark:hover:border-purple-700"
                        : "bg-card"
                    }`}
                  >
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <span className="inline-block w-1.5 h-1.5 bg-purple-500 rounded-full shrink-0"></span>
                      <span className="text-sm font-medium text-foreground truncate">
                        {detail?.toolset_name ?? toolsetId}
                      </span>
                      <span className="ml-1 px-1.5 py-0.5 text-[9px] font-semibold text-purple-600 bg-purple-50 border border-purple-200 rounded-sm uppercase tracking-wide shrink-0 dark:text-purple-300 dark:bg-purple-950 dark:border-purple-800">
                        Toolset
                      </span>
                    </div>
                    {toolCount > 0 && (
                      <div className="flex items-center gap-1 shrink-0 whitespace-nowrap">
                        <span className="text-xs font-medium text-muted-foreground">{toolCount}</span>
                        <span className="text-xs text-muted-foreground">{toolCount === 1 ? "tool" : "tools"}</span>
                        {isExpanded ? (
                          <ChevronDownIcon className="h-3.5 w-3.5 text-muted-foreground ml-0.5" />
                        ) : (
                          <ChevronRightIcon className="h-3.5 w-3.5 text-muted-foreground ml-0.5" />
                        )}
                      </div>
                    )}
                  </div>

                  {toolCount > 0 && isExpanded && detail && (
                    <div className="ml-4 pl-4 border-l-2 border-purple-200 pb-1">
                      <div className="flex flex-wrap gap-1.5">
                        {detail.tools.map((tool, toolIndex) => (
                          <span
                            key={toolIndex}
                            className="inline-flex items-center px-2.5 py-1 rounded-lg bg-purple-50 border border-purple-200 text-purple-800 text-xs font-medium dark:bg-purple-950 dark:border-purple-800 dark:text-purple-300"
                          >
                            <span className="text-purple-400 mr-1 text-[10px]">{tool.server_id.slice(0, 6)}…</span>
                            {tool.tool_name}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
        </div>
      ) : (
        <div className="flex items-center gap-2 px-3 py-2 rounded-lg bg-muted border border-border">
          <ServerIcon className="h-4 w-4 text-muted-foreground" />
          <p className="text-muted-foreground text-sm">No MCP servers, access groups, or toolsets configured</p>
        </div>
      )}
    </div>
  );
}

export default MCPServerPermissions;
