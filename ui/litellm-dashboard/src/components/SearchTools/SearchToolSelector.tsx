import React, { useEffect, useMemo, useState } from "react";
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
  /**
   * When set, only search tools whose IDs appear in this list are shown.
   * A list containing "*" means all tools are allowed (wildcard / legacy).
   * Undefined means no filtering (proxy admin without a team context).
   */
  allowedSearchToolIds?: string[];
}

const SearchToolSelector: React.FC<SearchToolSelectorProps> = ({
  onChange,
  value,
  className,
  accessToken,
  placeholder = "Select search tools",
  disabled = false,
  allowedSearchToolIds,
}) => {
  const [searchTools, setSearchTools] = useState<SearchTool[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const loadSearchTools = async () => {
      if (!accessToken) return;

      setLoading(true);
      try {
        const response = await fetchSearchTools(accessToken);
        if (response.search_tools) {
          setSearchTools(response.search_tools);
        }
      } catch (error) {
        console.error("Error fetching search tools:", error);
      } finally {
        setLoading(false);
      }
    };

    loadSearchTools();
  }, [accessToken]);

  // Filter tools based on team permissions
  const filteredTools = useMemo(() => {
    if (allowedSearchToolIds === undefined) return searchTools;
    // Wildcard means all tools are allowed
    if (allowedSearchToolIds.length === 1 && allowedSearchToolIds[0] === "*") return searchTools;
    // Empty list means no tools are allowed
    if (allowedSearchToolIds.length === 0) return [];
    // Filter to only allowed IDs
    return searchTools.filter(
      (tool) => allowedSearchToolIds.includes(tool.search_tool_id || tool.search_tool_name),
    );
  }, [searchTools, allowedSearchToolIds]);

  return (
    <div>
      <Select
        mode="multiple"
        placeholder={placeholder}
        onChange={onChange}
        value={value}
        loading={loading}
        className={className}
        allowClear
        options={filteredTools.map((tool) => ({
          label: `${tool.search_tool_name}${tool.search_tool_id ? ` (${tool.search_tool_id})` : ""}`,
          value: tool.search_tool_id || tool.search_tool_name,
          title: tool.search_tool_info?.description || tool.search_tool_name,
        }))}
        optionFilterProp="label"
        showSearch
        style={{ width: "100%" }}
        disabled={disabled}
      />
    </div>
  );
};

export default SearchToolSelector;
