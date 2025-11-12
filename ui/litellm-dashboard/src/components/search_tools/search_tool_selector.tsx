import React, { useEffect, useState } from "react";
import { Select } from "antd";
import { SearchTool } from "./types";
import { fetchSearchTools } from "../networking";

interface SearchToolSelectorProps {
  onChange: (selectedSearchTools: string[]) => void;
  value?: string[];
  className?: string;
  accessToken: string;
  placeholder?: string;
  disabled?: boolean;
}

const SearchToolSelector: React.FC<SearchToolSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select search tools",
  disabled = false,
}) => {
  const [searchTools, setSearchTools] = useState<SearchTool[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const fetchSearchToolList = async () => {
      if (!accessToken) return;

      setLoading(true);
      try {
        const response = await fetchSearchTools(accessToken);
        if (response.data) {
          setSearchTools(response.data);
        }
      } catch (error) {
        console.error("Error fetching search tools:", error);
      } finally {
        setLoading(false);
      }
    };

    fetchSearchToolList();
  }, [accessToken]);

  return (
    <div>
      <Select
        mode="multiple"
        placeholder={placeholder}
        onChange={onChange}
        value={value}
        className={className}
        disabled={disabled || loading}
        loading={loading}
        style={{ width: "100%" }}
      >
        {searchTools.map((tool) => (
          <Select.Option key={tool.search_tool_id} value={tool.search_tool_name}>
            {tool.search_tool_name}
          </Select.Option>
        ))}
      </Select>
    </div>
  );
};

export default SearchToolSelector;
