import { useMCPAccessGroups } from "@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { MultiSelect, type MultiSelectOption } from "@/components/shared/MultiSelect";
import React from "react";
import { ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL } from "@/components/mcp_tools/constants";
import type { MCPServer } from "@/components/mcp_tools/types";
import { accessGroupNamesOf } from "./effectiveMcpServers";

interface MCPServerSelectorProps {
  id?: string;
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
  userId?: string | null;
  allowNoMcpServers?: boolean;
  allowAllProxyMcpServers?: boolean;
}

const TOOLSET_PREFIX = "toolset:";
const ACCESS_GROUP_PREFIX = "access-group:";
const SERVER_PREFIX = "server:";
const SENTINELS: readonly string[] = [ALL_PROXY_MCP_SERVERS_SENTINEL, NO_MCP_SERVERS_SENTINEL];

const serverValue = (server: string): string => (SENTINELS.includes(server) ? server : `${SERVER_PREFIX}${server}`);

const valuesWithPrefix = (values: readonly string[], prefix: string): string[] =>
  values.filter((value) => value.startsWith(prefix)).map((value) => value.slice(prefix.length));

export const serversWithinAccessGroups = (
  servers: readonly MCPServer[],
  selectedAccessGroups: readonly string[],
  selectedServers: readonly string[],
): readonly MCPServer[] => {
  if (selectedAccessGroups.length === 0) return servers;
  return servers.filter(
    (server) =>
      server.allow_all_keys === true ||
      selectedServers.includes(server.server_id) ||
      accessGroupNamesOf(server).some((group) => selectedAccessGroups.includes(group)),
  );
};

const MCPServerSelector: React.FC<MCPServerSelectorProps> = ({
  id,
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select MCP servers",
  disabled = false,
  teamId,
  userId,
  allowNoMcpServers = false,
  allowAllProxyMcpServers = false,
}) => {
  const { data: mcpServers = [], isLoading: serversLoading } = useMCPServers(teamId, userId);
  const { data: accessGroups = [], isLoading: groupsLoading } = useMCPAccessGroups();
  const { data: toolsets = [], isLoading: toolsetsLoading } = useMCPToolsets();

  const loading = serversLoading || groupsLoading || toolsetsLoading;

  const visibleServers = serversWithinAccessGroups(mcpServers, value?.accessGroups ?? [], value?.servers ?? []);

  // Combine options: access groups + servers + toolsets
  const options = [
    ...accessGroups.map((group) => ({
      label: group,
      value: `${ACCESS_GROUP_PREFIX}${group}`,
      description: "Access Group",
    })),
    ...visibleServers.map((server) => ({
      label: `${server.server_name || server.server_id} (${server.server_id})`,
      value: serverValue(server.server_id),
      description: "MCP Server",
    })),
    ...toolsets.map((toolset) => ({
      label: toolset.toolset_name,
      value: `${TOOLSET_PREFIX}${toolset.toolset_id}`,
      description: "Toolset",
    })),
  ];

  const selectedValues = [
    ...(value?.servers || []).map(serverValue),
    ...(value?.accessGroups || []).map((group) => `${ACCESS_GROUP_PREFIX}${group}`),
    ...(value?.toolsets || []).map((id) => `${TOOLSET_PREFIX}${id}`),
  ];

  const listedValues = new Set(options.map((option) => option.value));
  const unlistedSelections = [
    ...(value?.servers || [])
      .filter((server) => !SENTINELS.includes(server))
      .map((server) => ({ label: server, value: serverValue(server), description: "MCP Server" })),
    ...(value?.accessGroups || []).map((group) => ({
      label: group,
      value: `${ACCESS_GROUP_PREFIX}${group}`,
      description: "Access Group",
    })),
  ].filter((option) => !listedValues.has(option.value));

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
    onChange({
      servers: [...selected.filter((v) => SENTINELS.includes(v)), ...valuesWithPrefix(selected, SERVER_PREFIX)],
      accessGroups: valuesWithPrefix(selected, ACCESS_GROUP_PREFIX),
      toolsets: valuesWithPrefix(selected, TOOLSET_PREFIX),
    });
  };

  const selectOptions: MultiSelectOption[] = [
    ...(allowAllProxyMcpServers || hasAllProxyMcpServersSelected
      ? [{ label: "All Proxy MCP Servers", value: ALL_PROXY_MCP_SERVERS_SENTINEL }]
      : []),
    ...(allowNoMcpServers
      ? [{ label: "No MCP Servers", value: NO_MCP_SERVERS_SENTINEL, description: "Block all" }]
      : []),
    ...[...options, ...unlistedSelections].map((opt) => ({
      ...opt,
      disabled: hasNoMcpServersSelected || hasAllProxyMcpServersSelected,
    })),
  ];

  return (
    <div>
      <MultiSelect
        id={id}
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
