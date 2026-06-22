import { Select } from "antd";
import React from "react";

import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";

interface LoggingExportersSelectProps {
  value?: string[];
  onChange?: (value: string[]) => void;
}

/**
 * Multi-select of admin-owned logging destinations (credential_type=logging) that an
 * identity (key / team / org) exports its traces to. The selected names are stored in
 * metadata.logging_exporters; the proxy unions them across the identity chain and fans
 * out. Sourced from the same registry, filtered to logging credentials only.
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
      notFoundContent="No logging destinations. Add one under Settings -> Logging Callbacks."
    />
  );
};

export default LoggingExportersSelect;
