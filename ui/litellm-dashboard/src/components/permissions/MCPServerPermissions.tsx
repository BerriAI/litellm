import React, { useState, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ChevronDown, ChevronRight, Server } from "lucide-react";
import { fetchMCPServers, fetchMCPToolsets } from "../networking";
import { MCPServer, MCPToolset } from "../mcp_tools/types";

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
  const [expandedServers, setExpandedServers] = useState<Set<string>>(
    new Set(),
  );
  const [expandedToolsets, setExpandedToolsets] = useState<Set<string>>(
    new Set(),
  );

  const toggle = (
    setter: React.Dispatch<React.SetStateAction<Set<string>>>,
    id: string,
  ) => {
    setter((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

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

  useEffect(() => {
    const fetchToolsets = async () => {
      if (accessToken && mcpToolsets.length > 0) {
        try {
          const all = await fetchMCPToolsets(accessToken);
          const filtered = Array.isArray(all)
            ? all.filter((t: MCPToolset) => mcpToolsets.includes(t.toolset_id))
            : [];
          setToolsetDetails(filtered);
        } catch (error) {
          console.error("Error fetching toolsets:", error);
        }
      }
    };
    fetchToolsets();
  }, [accessToken, mcpToolsets.length]);

  const getMCPServerDisplayName = (serverId: string) => {
    const serverDetail = mcpServerDetails.find(
      (s) => s.server_id === serverId,
    );
    if (serverDetail) {
      const truncatedId =
        serverId.length > 7
          ? `${serverId.slice(0, 3)}...${serverId.slice(-4)}`
          : serverId;
      return `${serverDetail.alias} (${truncatedId})`;
    }
    return serverId;
  };

  const mergedItems = [
    ...mcpServers.map((server) => ({ type: "server", value: server })),
    ...mcpAccessGroups.map((group) => ({ type: "accessGroup", value: group })),
  ];
  const totalCount = mergedItems.length + mcpToolsets.length;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Server className="h-4 w-4 text-blue-600 dark:text-blue-400" />
        <span className="font-semibold text-foreground">MCP Servers</span>
        <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300 text-xs">
          {totalCount}
        </Badge>
      </div>

      {totalCount > 0 ? (
        <div className="max-h-[400px] overflow-y-auto space-y-2 pr-1">
          {mergedItems.map((item, index) => {
            const toolsForServer =
              item.type === "server"
                ? mcpToolPermissions[item.value]
                : undefined;
            const hasToolRestrictions =
              toolsForServer && toolsForServer.length > 0;
            const isExpanded = expandedServers.has(item.value);

            return (
              <div key={index} className="space-y-2">
                <div
                  onClick={() =>
                    hasToolRestrictions &&
                    toggle(setExpandedServers, item.value)
                  }
                  className={`flex items-center gap-3 py-2 px-3 rounded-lg border border-border transition-all ${
                    hasToolRestrictions
                      ? "cursor-pointer hover:bg-muted hover:border-foreground/30"
                      : "bg-background"
                  }`}
                >
                  <div className="flex items-center gap-2 flex-1 min-w-0">
                    {item.type === "server" ? (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger asChild>
                            <div className="inline-flex items-center gap-2 min-w-0">
                              <span className="inline-block w-1.5 h-1.5 bg-blue-500 rounded-full flex-shrink-0" />
                              <span className="text-sm font-medium text-foreground truncate">
                                {getMCPServerDisplayName(item.value)}
                              </span>
                            </div>
                          </TooltipTrigger>
                          <TooltipContent>
                            Full ID: {item.value}
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    ) : (
                      <div className="inline-flex items-center gap-2 min-w-0">
                        <span className="inline-block w-1.5 h-1.5 bg-emerald-500 rounded-full flex-shrink-0" />
                        <span className="text-sm font-medium text-foreground truncate">
                          {item.value}
                        </span>
                        <span className="ml-1 px-1.5 py-0.5 text-[9px] font-semibold text-emerald-600 bg-emerald-50 border border-emerald-200 dark:bg-emerald-950/30 dark:border-emerald-900 dark:text-emerald-300 rounded uppercase tracking-wide flex-shrink-0">
                          Group
                        </span>
                      </div>
                    )}
                  </div>

                  {hasToolRestrictions && (
                    <div className="flex items-center gap-1 flex-shrink-0 whitespace-nowrap">
                      <span className="text-xs font-medium text-foreground">
                        {toolsForServer.length}
                      </span>
                      <span className="text-xs text-muted-foreground">
                        {toolsForServer.length === 1 ? "tool" : "tools"}
                      </span>
                      {isExpanded ? (
                        <ChevronDown className="h-3.5 w-3.5 text-muted-foreground ml-0.5" />
                      ) : (
                        <ChevronRight className="h-3.5 w-3.5 text-muted-foreground ml-0.5" />
                      )}
                    </div>
                  )}
                </div>

                {hasToolRestrictions && isExpanded && (
                  <div className="ml-4 pl-4 border-l-2 border-blue-200 dark:border-blue-900 pb-1">
                    <div className="flex flex-wrap gap-1.5">
                      {toolsForServer.map((tool, toolIndex) => (
                        <span
                          key={toolIndex}
                          className="inline-flex items-center px-2.5 py-1 rounded-lg bg-blue-50 border border-blue-200 text-blue-800 dark:bg-blue-950/30 dark:border-blue-900 dark:text-blue-200 text-xs font-medium"
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

          {mcpToolsets.length > 0 &&
            mcpToolsets.map((toolsetId, index) => {
              const detail = toolsetDetails.find(
                (t) => t.toolset_id === toolsetId,
              );
              const isExpanded = expandedToolsets.has(toolsetId);
              const toolCount = detail?.tools.length ?? 0;

              return (
                <div key={`toolset-${index}`} className="space-y-2">
                  <div
                    onClick={() =>
                      toolCount > 0 && toggle(setExpandedToolsets, toolsetId)
                    }
                    className={`flex items-center gap-3 py-2 px-3 rounded-lg border border-purple-200 dark:border-purple-900 transition-all ${
                      toolCount > 0
                        ? "cursor-pointer hover:bg-purple-50 dark:hover:bg-purple-950/30 hover:border-purple-300 dark:hover:border-purple-800"
                        : "bg-background"
                    }`}
                  >
                    <div className="flex items-center gap-2 flex-1 min-w-0">
                      <span className="inline-block w-1.5 h-1.5 bg-purple-500 rounded-full flex-shrink-0" />
                      <span className="text-sm font-medium text-foreground truncate">
                        {detail?.toolset_name ?? toolsetId}
                      </span>
                      <span className="ml-1 px-1.5 py-0.5 text-[9px] font-semibold text-purple-600 bg-purple-50 border border-purple-200 dark:bg-purple-950/30 dark:border-purple-900 dark:text-purple-300 rounded uppercase tracking-wide flex-shrink-0">
                        Toolset
                      </span>
                    </div>
                    {toolCount > 0 && (
                      <div className="flex items-center gap-1 flex-shrink-0 whitespace-nowrap">
                        <span className="text-xs font-medium text-foreground">
                          {toolCount}
                        </span>
                        <span className="text-xs text-muted-foreground">
                          {toolCount === 1 ? "tool" : "tools"}
                        </span>
                        {isExpanded ? (
                          <ChevronDown className="h-3.5 w-3.5 text-muted-foreground ml-0.5" />
                        ) : (
                          <ChevronRight className="h-3.5 w-3.5 text-muted-foreground ml-0.5" />
                        )}
                      </div>
                    )}
                  </div>

                  {toolCount > 0 && isExpanded && detail && (
                    <div className="ml-4 pl-4 border-l-2 border-purple-200 dark:border-purple-900 pb-1">
                      <div className="flex flex-wrap gap-1.5">
                        {detail.tools.map((tool, toolIndex) => (
                          <span
                            key={toolIndex}
                            className="inline-flex items-center px-2.5 py-1 rounded-lg bg-purple-50 border border-purple-200 text-purple-800 dark:bg-purple-950/30 dark:border-purple-900 dark:text-purple-200 text-xs font-medium"
                          >
                            <span className="text-purple-400 dark:text-purple-500 mr-1 text-[10px]">
                              {tool.server_id.slice(0, 6)}…
                            </span>
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
          <Server className="h-4 w-4 text-muted-foreground" />
          <span className="text-muted-foreground text-sm">
            No MCP servers, access groups, or toolsets configured
          </span>
        </div>
      )}
    </div>
  );
}

export default MCPServerPermissions;
