import React, { useEffect, useState } from 'react';
import { Select } from 'antd';
import { fetchMCPServers } from '../networking';
import { MCPServer } from '../mcp_tools/types';


interface MCPServerSelectorProps {
  onChange: (selectedMCPServers: string[]) => void;
  value?: string[];
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
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchMCPServerList = async () => {
      if (!accessToken) return;
      
      setLoading(true);
      try {
        const response = await fetchMCPServers(accessToken);
        if (response && Array.isArray(response)) {
          // Direct array response
          setMCPServers(response);
        } else if (response.data && Array.isArray(response.data)) {
          // Response with data wrapper
          setMCPServers(response.data);
        }
      } catch (error) {
        console.error("Error fetching MCP servers:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchMCPServerList();
  }, [accessToken]);

  return (
    <div>
      <Select
        mode="multiple"
        placeholder={placeholder}
        onChange={onChange}
        value={value}
        loading={loading}
        className={className}
        options={mcpServers.map(server => ({
          label: `${server.alias} (${server.server_id})`,
          value: server.server_id,
          title: server.description || server.alias,
        }))}
        optionFilterProp="label"
        showSearch
        style={{ width: '100%' }}
        disabled={disabled}
      />
    </div>
  );
};

export default MCPServerSelector; 