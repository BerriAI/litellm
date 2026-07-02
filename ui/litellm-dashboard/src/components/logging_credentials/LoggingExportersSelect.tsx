import { Select } from "antd";
import React from "react";

import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";

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
 * The options are exactly what GET /credentials returns for the caller, which the backend
 * already scopes: a proxy admin receives every destination, while a team or org admin
 * receives only the destinations granted to a scope they administer. Visibility is
 * enforced server-side by the same predicate the assignment gate and the request-time
 * resolver use, so this component does no role-based filtering of its own; doing so would
 * risk disagreeing with the backend in either direction.
 */
const LoggingExportersSelect: React.FC<LoggingExportersSelectProps> = ({ value, onChange }) => {
  const { data } = useCredentials();

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
      notFoundContent="No logging destinations available. Ask your proxy admin to create one under Settings -> Logging Callbacks."
    />
  );
};

export default LoggingExportersSelect;
