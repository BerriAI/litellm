import React from "react";
import { Select } from "antd";
import { Organization } from "../networking";

interface OrganizationDropdownProps {
  organizations?: Organization[] | null;
  value?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
  loading?: boolean;
}

const OrganizationDropdown: React.FC<OrganizationDropdownProps> = ({
  organizations,
  value,
  onChange,
  disabled,
  loading,
}) => {
  return (
    <Select
      showSearch
      placeholder="Search or select an organization"
      value={value}
      onChange={onChange}
      disabled={disabled}
      loading={loading}
      allowClear
      filterOption={(input, option) => {
        if (!option) return false;
        const org = organizations?.find((o) => o.organization_id === option.key);
        if (!org) return false;

        const searchTerm = input.toLowerCase().trim();
        const orgAlias = (org.organization_alias || "").toLowerCase();
        const orgId = (org.organization_id || "").toLowerCase();

        return orgAlias.includes(searchTerm) || orgId.includes(searchTerm);
      }}
      optionFilterProp="children"
    >
      {organizations?.map((org) => (
        <Select.Option key={org.organization_id} value={org.organization_id}>
          <span className="font-medium">{org.organization_alias}</span>{" "}
          <span className="text-gray-500">({org.organization_id})</span>
        </Select.Option>
      ))}
    </Select>
  );
};

export default OrganizationDropdown;
