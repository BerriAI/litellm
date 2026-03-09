import React from "react";
import { Select, Skeleton } from "antd";
import { TeamOutlined } from "@ant-design/icons";
import { Text } from "@tremor/react";
import {
  useAccessGroups,
  AccessGroupResponse,
} from "@/app/(dashboard)/hooks/accessGroups/useAccessGroups";

export interface AccessGroupSelectorProps {
  value?: string[];
  onChange?: (value: string[]) => void;
  placeholder?: string;
  disabled?: boolean;
  style?: React.CSSProperties;
  className?: string;
  showLabel?: boolean;
  labelText?: string;
  /** Allow clearing the selection */
  allowClear?: boolean;
}

/**
 * Reusable multi-select selector for access groups.
 *
 * - Displays the **access_group_name** in the dropdown.
 * - Returns an array of **access_group_id** values.
 * - Always multi-select since users can assign multiple access groups.
 * - Integrates with Ant Design `<Form.Item>` out of the box via `value` / `onChange`.
 */
const AccessGroupSelector: React.FC<AccessGroupSelectorProps> = ({
  value,
  onChange,
  placeholder = "Select access groups",
  disabled = false,
  style,
  className,
  showLabel = false,
  labelText = "Access Group",
  allowClear = true,
}) => {
  const { data: accessGroups, isLoading, isError } = useAccessGroups();

  // ── Loading skeleton ─────────────────────────────────────────────────────
  if (isLoading) {
    return (
      <div>
        {showLabel && (
          <Text className="font-medium block mb-2 text-gray-700 flex items-center">
            <TeamOutlined className="mr-2" /> {labelText}
          </Text>
        )}
        <Skeleton.Input active block style={{ height: 32, ...style }} />
      </div>
    );
  }

  // ── Build options ────────────────────────────────────────────────────────
  const options = (accessGroups ?? []).map((group: AccessGroupResponse) => ({
    label: (
      <span>
        <span className="font-medium">{group.access_group_name}</span>{" "}
        <span className="text-gray-400 text-xs">({group.access_group_id})</span>
      </span>
    ),
    value: group.access_group_id,
    selectedLabel: group.access_group_name,
    searchText: `${group.access_group_name} ${group.access_group_id}`,
  }));

  // ── Render ───────────────────────────────────────────────────────────────
  return (
    <div>
      {showLabel && (
        <Text className="font-medium block mb-2 text-gray-700 flex items-center">
          <TeamOutlined className="mr-2" /> {labelText}
        </Text>
      )}
      <Select
        mode="multiple"
        value={value}
        placeholder={placeholder}
        onChange={onChange}
        disabled={disabled}
        allowClear={allowClear}
        showSearch
        style={{ width: "100%", ...style }}
        className={`rounded-md ${className ?? ""}`}
        notFoundContent={
          isError ? (
            <span className="text-red-500">Failed to load access groups</span>
          ) : (
            "No access groups found"
          )
        }
        filterOption={(input, option) => {
          const searchText =
            options.find((opt) => opt.value === option?.value)?.searchText ?? "";
          return searchText.toLowerCase().includes(input.toLowerCase());
        }}
        optionLabelProp="selectedLabel"
        options={options.map((opt) => ({
          label: opt.label,
          value: opt.value,
          selectedLabel: opt.selectedLabel,
        }))}
      />
    </div>
  );
};

export default AccessGroupSelector;
