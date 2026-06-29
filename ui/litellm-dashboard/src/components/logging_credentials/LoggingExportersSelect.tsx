import { Select } from "antd";
import React from "react";

import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { isAdminRole } from "@/utils/roles";

interface LoggingExportersSelectProps {
  value?: string[];
  onChange?: (value: string[]) => void;
}

/**
 * Multi-select of admin-owned logging destinations (credential_type=logging) that an
 * identity (key / team / org) exports its traces to. The selected names are stored in
 * metadata.logging_exporters; the proxy unions them across the identity chain and fans
 * out.
 *
 * Options are scoped to what the caller can actually assign: a proxy admin sees every
 * destination; everyone else sees only the ones visible to a team or org they belong to
 * (plus global / auto_enable destinations). This mirrors the backend assignment gate so
 * a team admin is not offered another tenant's destination only to have the save
 * rejected, and it avoids surfacing other tenants' destination names. The backend stays
 * the authoritative check -- this filter is UX, not a security boundary.
 */
const LoggingExportersSelect: React.FC<LoggingExportersSelectProps> = ({ value, onChange }) => {
  const { data } = useCredentials();
  const { userRole } = useAuthorized();
  const { data: teams } = useTeams();
  const { data: orgs } = useOrganizations();

  const seesEveryDestination = isAdminRole(userRole ?? "");
  const myTeamIds = new Set((teams ?? []).map((t) => t.team_id));
  const myOrgIds = new Set((orgs ?? []).map((o) => o.organization_id));

  const assignable = (info: {
    auto_enable?: boolean;
    access?: { global?: boolean; teams?: string[]; orgs?: string[] };
  }) => {
    if (info.auto_enable === true || info.access?.global === true) return true;
    if ((info.access?.teams ?? []).some((id) => myTeamIds.has(id))) return true;
    return (info.access?.orgs ?? []).some((id) => myOrgIds.has(id));
  };

  const options = (data?.credentials ?? [])
    .filter((credential) => credential.credential_info?.credential_type === "logging")
    .filter((credential) => seesEveryDestination || assignable(credential.credential_info))
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
