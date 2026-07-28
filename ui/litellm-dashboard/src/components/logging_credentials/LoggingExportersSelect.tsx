import { Form, Select } from "antd";
import React from "react";

import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isProxyAdminRole } from "@/utils/roles";

interface LoggingExportersSelectProps {
  value?: string[];
  onChange?: (value: string[]) => void;
}

/**
 * Multi-select of admin-owned logging destinations (credential_type=logging) that an
 * identity (key / team / org) exports its traces to. The selected names are persisted to
 * the identity's logging_exporters column; the proxy unions them across the identity
 * chain and fans out.
 *
 * Assigning logging exporters is proxy-admin only (GET /credentials and the assignment
 * gate both reject non-proxy-admins), so this control renders only for a proxy admin.
 */
const LoggingExportersSelect: React.FC<LoggingExportersSelectProps> = ({ value, onChange }) => {
  const { userRole } = useAuthorized();
  const isProxyAdmin = userRole ? isProxyAdminRole(userRole) : false;
  const { data } = useCredentials(isProxyAdmin);

  if (!isProxyAdmin) {
    return null;
  }

  const options = (data?.credentials ?? [])
    .filter((credential) => credential.credential_info?.credential_type === "logging")
    .map((credential) => ({
      value: credential.credential_name,
      label: credential.credential_info?.host
        ? `${credential.credential_name} (${credential.credential_info.host})`
        : credential.credential_name,
    }));

  return (
    <Select
      mode="multiple"
      allowClear
      placeholder="Select logging destinations this identity exports to"
      value={value}
      onChange={onChange}
      options={options}
      style={{ width: "100%" }}
      optionFilterProp="label"
      notFoundContent="No logging destinations available. Create one under Settings -> Logging Callbacks."
    />
  );
};

export default LoggingExportersSelect;

interface LoggingExportersFormItemProps {
  tooltip: string;
  className?: string;
}

/**
 * The antd Form.Item wrapper for LoggingExportersSelect, gated to proxy admins so
 * non-admin forms render neither the picker nor an orphaned "Logging Exporters"
 * label. Keeps the role gate in one place for every antd form that binds the
 * logging_exporters field.
 */
export const LoggingExportersFormItem: React.FC<LoggingExportersFormItemProps> = ({ tooltip, className }) => {
  const { userRole } = useAuthorized();
  if (userRole == null || !isProxyAdminRole(userRole)) {
    return null;
  }
  return (
    <Form.Item label="Logging Exporters" name="logging_exporters" tooltip={tooltip} className={className}>
      <LoggingExportersSelect />
    </Form.Item>
  );
};
