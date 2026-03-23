import { useMCPAccessGroups } from "@/app/(dashboard)/hooks/mcpServers/useMCPAccessGroups";
import { useMCPServers } from "@/app/(dashboard)/hooks/mcpServers/useMCPServers";
import { useMCPToolsets } from "@/app/(dashboard)/hooks/mcpServers/useMCPToolsets";
import { Select } from "antd";
import React from "react";

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

  // Handle selection
  const handleChange = (selected: string[]) => {
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
          const searchText = options.find((opt) => opt.value === option?.value)?.searchText || "";
          return searchText.toLowerCase().includes(input.toLowerCase());
        }}
      >
        {options.map((opt) => (
          <Select.Option key={opt.value} value={opt.value} label={opt.label}>
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
