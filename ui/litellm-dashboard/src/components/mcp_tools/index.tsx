import React, { useState } from 'react';
import { useQuery, useMutation } from '@tanstack/react-query';
import { DataTable } from '../view_logs/table';
import { columns, ToolTestPanel } from './columns';
import { MCPTool, MCPToolsViewerProps, CallMCPToolResponse } from './types';
import { listMCPTools, callMCPTool } from '../networking';
import MCPServers from './mcp_servers';

// Wrapper to handle the type mismatch between MCPTool and DataTable's expected type
function DataTableWrapper({
  columns,
  data,
  isLoading,
}: {
  columns: any;
  data: MCPTool[];
  isLoading: boolean;
}) {
  // Create a dummy renderSubComponent and getRowCanExpand function
  const renderSubComponent = () => <div />;
  const getRowCanExpand = () => false;

  return (
    <DataTable
      columns={columns as any}
      data={data as any}
      isLoading={isLoading}
      renderSubComponent={renderSubComponent}
      getRowCanExpand={getRowCanExpand}
      loadingMessage="🚅 Loading tools..."
      noDataMessage="No tools found"
    />
  );
}

const MCPToolsViewer = ({
  serverId,
  accessToken,
  userRole,
  userID,
}: MCPToolsViewerProps) => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedTool, setSelectedTool] = useState<MCPTool | null>(null);
  const [toolResult, setToolResult] = useState<CallMCPToolResponse | null>(null);
  const [toolError, setToolError] = useState<Error | null>(null);

  // Query to fetch MCP tools
  const { data: mcpTools, isLoading: isLoadingTools } = useQuery({
    queryKey: ['mcpTools'],
    queryFn: () => {
      if (!accessToken) throw new Error('Access Token required');
      return listMCPTools(accessToken, serverId);
    },
    enabled: !!accessToken,
  });

  // Mutation for calling a tool
  const { mutate: executeTool, isPending: isCallingTool } = useMutation({
    mutationFn: (args: { tool: MCPTool; arguments: Record<string, any> }) => {
      if (!accessToken) throw new Error('Access Token required');
      return callMCPTool(
        accessToken,
        args.tool.name,
        args.arguments
      );
    },
    onSuccess: (data) => {
      setToolResult(data);
      setToolError(null);
    },
    onError: (error: Error) => {
      setToolError(error);
      setToolResult(null);
    },
  });

  // Add onToolSelect handler to each tool
  const toolsData = React.useMemo(() => {
    if (!mcpTools) return [];
    
    return mcpTools.map((tool: MCPTool) => ({
      ...tool,
      onToolSelect: (tool: MCPTool) => {
        setSelectedTool(tool);
        setToolResult(null);
        setToolError(null);
      }
    }));
  }, [mcpTools]);

  // Filter tools based on search term
  const filteredTools = React.useMemo(() => {
    return toolsData.filter((tool: MCPTool) => {
      const searchLower = searchTerm.toLowerCase();
      return (
        tool.name.toLowerCase().includes(searchLower) ||
        (tool.description != null && tool.description.toLowerCase().includes(searchLower)) ||
        tool.mcp_info.server_name.toLowerCase().includes(searchLower)
      );
    });
  }, [toolsData, searchTerm]);

  // Handle tool call submission
  const handleToolSubmit = (args: Record<string, any>) => {
    if (!selectedTool) return;
    
    executeTool({
      tool: selectedTool,
      arguments: args,
    });
  };

  if (!accessToken || !userRole || !userID) {
    return <div className="p-6 text-center text-gray-500">Missing required authentication parameters.</div>;
  }

  return (
    <div className="w-full p-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-semibold">MCP Tools</h1>
      </div>

      <div className="bg-white rounded-lg shadow">
        <div className="border-b px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="relative w-64">
              <input
                type="text"
                placeholder="Search tools..."
                className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
              <svg
                className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
                fill="none"
                stroke="currentColor"
                viewBox="0 0 24 24"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                />
              </svg>
            </div>
            <div className="text-sm text-gray-500">
              {filteredTools.length} tool{filteredTools.length !== 1 ? "s" : ""} available
            </div>
          </div>
        </div>

        <DataTableWrapper
          columns={columns}
          data={filteredTools}
          isLoading={isLoadingTools}
        />
      </div>

      {/* Tool Test Panel - Show when a tool is selected */}
      {selectedTool && (
        <div className="fixed inset-0 bg-gray-800 bg-opacity-75 flex items-center justify-center z-50 p-4">
          <ToolTestPanel
            tool={selectedTool}
            onSubmit={handleToolSubmit}
            isLoading={isCallingTool}
            result={toolResult}
            error={toolError}
            onClose={() => setSelectedTool(null)}
          />
        </div>
      )}
    </div>
  );
}

export { MCPToolsViewer, MCPServers };