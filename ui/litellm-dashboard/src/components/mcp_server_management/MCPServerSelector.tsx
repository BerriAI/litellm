import { useMCPAccessGroups } from "@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { Select } from "antd";
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

  // Combine options: access groups (green) + servers (blue) + toolsets (purple)
  const options = [
    ...accessGroups.map((group) => ({
      label: group,
      value: group,
      type: "accessGroup" as const,
      searchText: `${group} Access Group`,
    })),
    ...mcpServers.map((server) => ({
      label: `${server.server_name || server.server_id} (${server.server_id})`,
      value: server.server_id,
      type: "server" as const,
      searchText: `${server.server_name || server.server_id} ${server.server_id} MCP Server`,
    })),
    ...toolsets.map((toolset) => ({
      label: toolset.toolset_name,
      value: `${TOOLSET_PREFIX}${toolset.toolset_id}`,
      type: "toolset" as const,
      searchText: `${toolset.toolset_name} ${toolset.toolset_id} Toolset`,
    })),
  ];

  const colorByType: Record<string, string> = {
    accessGroup: "#52c41a",
    server: "#1890ff",
    toolset: "#722ed1",
  };
  const labelByType: Record<string, string> = {
    accessGroup: "Access Group",
    server: "MCP Server",
    toolset: "Toolset",
  };

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

  return (
    <div>
      <Select
        mode="multiple"
        placeholder={placeholder}
        onChange={handleChange}
        value={selectedValues}
        loading={loading}
        className={className}
        allowClear
        showSearch
        style={{ width: "100%" }}
        disabled={disabled}
        filterOption={(input, option) => {
          if (option?.value === NO_MCP_SERVERS_SENTINEL) return true;
          if (option?.value === ALL_PROXY_MCP_SERVERS_SENTINEL) return true;
          const searchText = options.find((opt) => opt.value === option?.value)?.searchText || "";
          return searchText.toLowerCase().includes(input.toLowerCase());
        }}
      >
        {(allowAllProxyMcpServers || hasAllProxyMcpServersSelected) && (
          <Select.Option
            key={ALL_PROXY_MCP_SERVERS_SENTINEL}
            value={ALL_PROXY_MCP_SERVERS_SENTINEL}
            label="All Proxy MCP Servers"
          >
            <span style={{ color: "#1890ff", fontWeight: 500 }}>All Proxy MCP Servers</span>
          </Select.Option>
        )}
        {allowNoMcpServers && (
          <Select.Option key={NO_MCP_SERVERS_SENTINEL} value={NO_MCP_SERVERS_SENTINEL} label="No MCP Servers">
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span style={{ flex: 1 }}>No MCP Servers</span>
              <span style={{ color: "#8c8c8c", fontSize: "12px", fontWeight: 500, opacity: 0.8 }}>Block all</span>
            </div>
          </Select.Option>
        )}
        {options.map((opt) => (
          <Select.Option
            key={opt.value}
            value={opt.value}
            label={opt.label}
            disabled={hasNoMcpServersSelected || hasAllProxyMcpServersSelected}
          >
            <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
              <span
                style={{
                  display: "inline-block",
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: colorByType[opt.type],
                  flexShrink: 0,
                }}
              />
              <span style={{ flex: 1 }}>{opt.label}</span>
              <span
                style={{
                  color: colorByType[opt.type],
                  fontSize: "12px",
                  fontWeight: 500,
                  opacity: 0.8,
                }}
              >
                {labelByType[opt.type]}
              </span>
            </div>
          </Select.Option>
        ))}
      </Select>
    </div>
  );
};

export default MCPServerSelector;
