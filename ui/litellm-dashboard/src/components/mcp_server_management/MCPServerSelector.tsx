import React, { useEffect, useState } from 'react';
import { Select } from 'antd';
import { fetchMCPServers, fetchMCPAccessGroups } from '../networking';
import { MCPServer } from '../mcp_tools/types';

interface MCPServerSelectorProps {
  onChange: (selected: { servers: string[]; accessGroups: string[] }) => void;
  value?: { servers: string[]; accessGroups: string[] };
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
  disabled = false
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
          fetchMCPAccessGroups(accessToken)
        ]);
        let servers = Array.isArray(serversRes) ? serversRes : (serversRes.data || []);
        let groups = Array.isArray(groupsRes) ? groupsRes : (groupsRes.data || []);
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
    ...accessGroups.map(group => ({
      label: group,
      value: group,
      isAccessGroup: true
    })),
    ...mcpServers.map(server => ({
      label: `${server.alias || server.server_id} (${server.server_id})`,
      value: server.server_id,
      isAccessGroup: false
    }))
  ];

  // Flatten value for Select
  const selectedValues = [
    ...(value?.servers || []),
    ...(value?.accessGroups || [])
  ];

  // Handle selection
  const handleChange = (selected: string[]) => {
    const servers = selected.filter(v => !accessGroups.includes(v));
    const accessGroupsSelected = selected.filter(v => accessGroups.includes(v));
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
        optionFilterProp="label"
        showSearch
        style={{ width: '100%' }}
        disabled={disabled}
      >
        {options.map(opt => (
          <Select.Option
            key={opt.value}
            value={opt.value}
          >
            {opt.isAccessGroup && (
              <span style={{
                display: 'inline-block',
                width: 10,
                height: 10,
                borderRadius: '50%',
                background: '#1890ff',
                marginRight: 8,
                verticalAlign: 'middle',
              }} />
            )}
            {opt.label}
            {opt.isAccessGroup && <span style={{ color: '#1890ff', marginLeft: 8 }}>(Access Group)</span>}
          </Select.Option>
        ))}
      </Select>
    </div>
  );
};

export default MCPServerSelector; 