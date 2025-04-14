import React from "react";
import { Select } from "antd";
import { Team } from "../key_team_helpers/key_list";

interface GeneralDropdownProps {
  items?: any[] | null;
  value?: string;
  key_to_sort_by?: string;
  onChange?: (value: string) => void;
  placeholder?: string;
  clearable?: boolean;
  props?: any;
}

const GeneralDropdown: React.FC<GeneralDropdownProps> = ({ items, value, key_to_sort_by, onChange, placeholder = "Search", clearable = true, ...props }) => {
  return (
    <Select
      showSearch
      placeholder={placeholder}
      value={value}
      allowClear={clearable}
      onChange={onChange}
      filterOption={(input, option) => {
        if (!option) return false;
        const teamAlias = option.children?.[0]?.props?.children || '';
        return teamAlias.toLowerCase().includes(input.toLowerCase());
      }}
      optionFilterProp="children"
      {...props}
    >
      {items?.map((item) => (
        <Select.Option key={item[key_to_sort_by]} value={item[key_to_sort_by]}>
          <span className="font-medium">{item[key_to_sort_by]}</span>{" "}
          <span className="text-gray-500">({item[key_to_sort_by]})</span>
        </Select.Option>
      ))}
    </Select>
  );
};

export default GeneralDropdown; 