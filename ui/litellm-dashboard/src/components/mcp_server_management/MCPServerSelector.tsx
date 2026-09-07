import { useMCPAccessGroups } from "@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { MultiSelect, type MultiSelectOption } from "@/components/shared/MultiSelect";
import React from "react";
import { ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";

interface MCPServerSelectorProps {
  onChange: (selected: { servers: string[]; accessGroups: string[]; toolsets: string[] }) => void;
  value?: {
    servers: string[];
    accessGroups: string[];
    toolsets?: string[];
  };
  className?: string;
  accessToken: string;
  placeholder?: string;
  disabled?: boolean;
  teamId?: string | null;
  allowNoMcpServers?: boolean;
  allowAllProxyMcpServers?: boolean;
}

const TOOLSET_PREFIX = "toolset:";

const MCPServerSelector: React.FC<MCPServerSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select MCP servers",
  disabled = false,
  teamId,
  allowNoMcpServers = false,
  allowAllProxyMcpServers = false,
}) => {
  const { data: mcpServers = [], isLoading: serversLoading } = useMCPServers(teamId);
  const { data: accessGroups = [], isLoading: groupsLoading } = useMCPAccessGroups();
  const { data: toolsets = [], isLoading: toolsetsLoading } = useMCPToolsets();

  const loading = serversLoading || groupsLoading || toolsetsLoading;

  const accessGroupSet = new Set(accessGroups);

  // Combine options: access groups + servers + toolsets
  const options = [
    ...accessGroups.map((group) => ({
      label: group,
      value: group,
      description: "Access Group",
    })),
    ...mcpServers.map((server) => ({
      label: `${server.server_name || server.server_id} (${server.server_id})`,
      value: server.server_id,
      description: "MCP Server",
    })),
    ...toolsets.map((toolset) => ({
      label: toolset.toolset_name,
      value: `${TOOLSET_PREFIX}${toolset.toolset_id}`,
      description: "Toolset",
    })),
  ];

  // Flatten value for Select — prefix toolset IDs
  const selectedValues = [
    ...(value?.servers || []),
    ...(value?.accessGroups || []),
    ...(value?.toolsets || []).map((id) => `${TOOLSET_PREFIX}${id}`),
  ];

  const hasNoMcpServersSelected = allowNoMcpServers && selectedValues.includes(NO_MCP_SERVERS_SENTINEL);
  const hasAllProxyMcpServersSelected = selectedValues.includes(ALL_PROXY_MCP_SERVERS_SENTINEL);

  // Handle selection
  const handleChange = (selected: string[]) => {
    if (allowAllProxyMcpServers && selected.includes(ALL_PROXY_MCP_SERVERS_SENTINEL)) {
      onChange({ servers: [ALL_PROXY_MCP_SERVERS_SENTINEL], accessGroups: [], toolsets: [] });
      return;
    }
    // "No MCP Servers" is exclusive: picking it clears everything else.
    if (allowNoMcpServers && selected.includes(NO_MCP_SERVERS_SENTINEL)) {
      onChange({ servers: [NO_MCP_SERVERS_SENTINEL], accessGroups: [], toolsets: [] });
      return;
    }
    const toolsetsSelected = selected
      .filter((v) => v.startsWith(TOOLSET_PREFIX))
      .map((v) => v.slice(TOOLSET_PREFIX.length));
    const rest = selected.filter((v) => !v.startsWith(TOOLSET_PREFIX));
    const servers = rest.filter((v) => !accessGroupSet.has(v));
    const accessGroupsSelected = rest.filter((v) => accessGroupSet.has(v));
    onChange({ servers, accessGroups: accessGroupsSelected, toolsets: toolsetsSelected });
  };

  const selectOptions: MultiSelectOption[] = [
    ...(allowAllProxyMcpServers || hasAllProxyMcpServersSelected
      ? [{ label: "All Proxy MCP Servers", value: ALL_PROXY_MCP_SERVERS_SENTINEL }]
      : []),
    ...(allowNoMcpServers
      ? [{ label: "No MCP Servers", value: NO_MCP_SERVERS_SENTINEL, description: "Block all" }]
      : []),
    ...options.map((opt) => ({
      ...opt,
      disabled: hasNoMcpServersSelected || hasAllProxyMcpServersSelected,
    })),
  ];

  return (
    <div>
      <MultiSelect
        options={selectOptions}
        value={selectedValues}
        onValueChange={handleChange}
        placeholder={placeholder}
        emptyText="No MCP servers found"
        loading={loading}
        disabled={disabled}
        className={`w-full ${className ?? ""}`}
      />
    </div>
  );
};

export default MCPServerSelector;
