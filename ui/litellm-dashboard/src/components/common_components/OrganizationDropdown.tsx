import React from "react";
import { Select, Typography } from "antd";
import { Organization } from "../networking";

const { Text } = Typography;

interface OrganizationDropdownProps {
  organizations?: Organization[] | null;
  value?: string;
  onChange?: (value: string) => void;
  disabled?: boolean;
  loading?: boolean;
  style?: React.CSSProperties;
}

const OrganizationDropdown: React.FC<OrganizationDropdownProps> = ({
  organizations,
  value,
  onChange,
  disabled,
  loading,
  style,
}) => {
  return (
    <Select
      showSearch
      placeholder="All Organizations"
      value={value}
      onChange={onChange}
      disabled={disabled}
      loading={loading}
      allowClear
      style={{ minWidth: 280, ...style }}
      filterOption={(input, option) => {
        if (!option) return false;
        const org = organizations?.find((o) => o.organization_id === option.key);
        if (!org) return false;

        const searchTerm = input.toLowerCase().trim();
        const orgAlias = (org.organization_alias || "").toLowerCase();
        const orgId = (org.organization_id || "").toLowerCase();

        return orgAlias.includes(searchTerm) || orgId.includes(searchTerm);
      }}
    >
      {organizations?.map((org) => (
        <Select.Option key={org.organization_id} value={org.organization_id}>
          <span className="font-medium">{org.organization_alias}</span>{" "}
          <Text type="secondary">({org.organization_id})</Text>
        </Select.Option>
      ))}
    </Select>
  );
};

export default OrganizationDropdown;
