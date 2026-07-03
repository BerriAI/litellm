import { useMCPAccessGroups } from "@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { Select } from "antd";
import React from "react";
import {
  ALL_PROXY_MCPS_SENTINEL,
  ALL_TEAM_MCPS_SENTINEL,
  NO_MCP_SERVERS_SENTINEL,
} from "@/components/mcp_tools/constants";

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
  allowAllProxyMcps?: boolean;
  allowAllTeamMcps?: boolean;
}

const TOOLSET_PREFIX = "toolset:";

type ExclusiveSentinel = { value: string; label: string; caption: string; color: string };

// Each entry grants a whole class of servers, so it is mutually exclusive with
// every other pick. Ordered most-restrictive first to mirror the backend grant
// precedence (block-all > team > proxy); the first match wins. "All Team MCPs"
// only appears once a team is chosen, since it resolves against that team at
// request time and is meaningless on a teamless key.
const buildExclusiveSentinels = (
  allowNoMcpServers: boolean,
  allowAllTeamMcps: boolean,
  allowAllProxyMcps: boolean,
  teamId: string | null | undefined,
): ExclusiveSentinel[] => [
  ...(allowNoMcpServers
    ? [{ value: NO_MCP_SERVERS_SENTINEL, label: "No MCP Servers", caption: "Block all", color: "#8c8c8c" }]
    : []),
  ...(allowAllTeamMcps && !!teamId
    ? [{ value: ALL_TEAM_MCPS_SENTINEL, label: "All Team MCPs", caption: "Team's servers", color: "#13c2c2" }]
    : []),
  ...(allowAllProxyMcps
    ? [{ value: ALL_PROXY_MCPS_SENTINEL, label: "All Proxy MCPs", caption: "Every server", color: "#1890ff" }]
    : []),
];

const MCPServerSelector: React.FC<MCPServerSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select MCP servers",
  disabled = false,
  teamId,
  allowNoMcpServers = false,
  allowAllProxyMcps = false,
  allowAllTeamMcps = false,
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

  const exclusiveSentinels = buildExclusiveSentinels(allowNoMcpServers, allowAllTeamMcps, allowAllProxyMcps, teamId);

  const selectedExclusive = exclusiveSentinels.find((s) => selectedValues.includes(s.value)) ?? null;
  const hasExclusiveSelected = selectedExclusive !== null;

  // Handle selection
  const handleChange = (selected: string[]) => {
    const exclusive = exclusiveSentinels.find((s) => selected.includes(s.value));
    if (exclusive) {
      onChange({ servers: [exclusive.value], accessGroups: [], toolsets: [] });
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

  const renderExclusiveOptions = () =>
    exclusiveSentinels.map((sentinel) => (
      <Select.Option
        key={sentinel.value}
        value={sentinel.value}
        label={sentinel.label}
        disabled={hasExclusiveSelected && selectedExclusive?.value !== sentinel.value}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "8px" }}>
          <span style={{ flex: 1 }}>{sentinel.label}</span>
          <span style={{ color: sentinel.color, fontSize: "12px", fontWeight: 500, opacity: 0.8 }}>
            {sentinel.caption}
          </span>
        </div>
      </Select.Option>
    ));

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
          if (exclusiveSentinels.some((s) => s.value === option?.value)) return true;
          const searchText = options.find((opt) => opt.value === option?.value)?.searchText || "";
          return searchText.toLowerCase().includes(input.toLowerCase());
        }}
      >
        {renderExclusiveOptions()}
        {options.map((opt) => (
          <Select.Option key={opt.value} value={opt.value} label={opt.label} disabled={hasExclusiveSelected}>
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
