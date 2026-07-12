import React, { useEffect, useState } from "react";
import { Select } from "antd";
import { fetchSearchTools } from "../networking";

export interface SearchToolSelectorProps {
  onChange: (selected: string[]) => void;
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
  placeholder = "Select search tools (optional)",
  disabled = false,
}) => {
  const [options, setOptions] = useState<{ label: string; value: string }[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const load = async () => {
      if (!accessToken) return;
      setLoading(true);
      try {
        const data = await fetchSearchTools(accessToken);
        const tools = Array.isArray(data?.search_tools)
          ? data.search_tools
          : Array.isArray(data?.data)
            ? data.data
            : [];
        setOptions(
          tools
            .map((tool: { search_tool_name?: string }) => tool?.search_tool_name)
            .filter((name: unknown): name is string => typeof name === "string" && name.length > 0)
            .map((name: string) => ({ label: name, value: name })),
        );
      } catch (e) {
        console.error("Failed to load search tools:", e);
      } finally {
        setLoading(false);
      }
    };
    load();
  }, [accessToken]);

  return (
    <Select
      mode="multiple"
      allowClear
      showSearch
      optionFilterProp="label"
      placeholder={placeholder}
      onChange={onChange}
      value={value}
      loading={loading}
      className={className}
      options={options}
      style={{ width: "100%" }}
      disabled={disabled}
    />
  );
};

export default SearchToolSelector;
