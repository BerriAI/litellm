import React, { useEffect, useState } from "react";
import { Select } from "antd";
import { fetchMCPServers, fetchMCPAccessGroups } from "../networking";
import { MCPServer } from "../mcp_tools/types";

interface MCPServerSelectorProps {
  onChange: (selected: { 
    servers: string[]; 
    accessGroups: string[];
  }) => void;
  value?: { 
    servers: string[]; 
    accessGroups: string[];
  };
  className?: string;
  accessToken: string;
  placeholder?: string;
  disabled?: boolean;
}

const MCPServerSelector: React.FC<MCPServerSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select MCP servers",
  disabled = false,
}) => {
  const [mcpServers, setMCPServers] = useState<MCPServer[]>([]);
  const [accessGroups, setAccessGroups] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      if (!accessToken) return;
      setLoading(true);
      try {
        const [serversRes, groupsRes] = await Promise.all([
          fetchMCPServers(accessToken),
          fetchMCPAccessGroups(accessToken),
        ]);
        let servers = Array.isArray(serversRes) ? serversRes : serversRes.data || [];
        let groups = Array.isArray(groupsRes) ? groupsRes : groupsRes.data || [];
        setMCPServers(servers);
        setAccessGroups(groups);
      } catch (error) {
        console.error("Error fetching MCP servers or access groups:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, [accessToken]);

  // Combine options, access groups first
  const options = [
    ...accessGroups.map((group) => ({
      label: group,
      value: group,
      isAccessGroup: true,
      searchText: `${group} Access Group`,
    })),
    ...mcpServers.map((server) => ({
      label: `${server.server_name || server.server_id} (${server.server_id})`,
      value: server.server_id,
      isAccessGroup: false,
      searchText: `${server.server_name || server.server_id} ${server.server_id} MCP Server`,
    })),
  ];

  // Flatten value for Select
  const selectedValues = [...(value?.servers || []), ...(value?.accessGroups || [])];

  // Handle selection
  const handleChange = (selected: string[]) => {
    const servers = selected.filter((v) => !accessGroups.includes(v));
    const accessGroupsSelected = selected.filter((v) => accessGroups.includes(v));
    onChange({ servers, accessGroups: accessGroupsSelected });
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
                  background: opt.isAccessGroup ? "#52c41a" : "#1890ff",
                  flexShrink: 0,
                }}
              />
              <span style={{ flex: 1 }}>{opt.label}</span>
              <span
                style={{
                  color: opt.isAccessGroup ? "#52c41a" : "#1890ff",
                  fontSize: "12px",
                  fontWeight: 500,
                  opacity: 0.8,
                }}
              >
                {opt.isAccessGroup ? "Access Group" : "MCP Server"}
              </span>
            </div>
          </Select.Option>
        ))}
      </Select>
    </div>
  );
};

export default MCPServerSelector;
