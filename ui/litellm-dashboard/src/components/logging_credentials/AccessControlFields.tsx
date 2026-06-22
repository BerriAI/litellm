import { Form, Select, Switch } from "antd";
import React from "react";

import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { CredentialAccess } from "../Settings/LoggingAndAlerts/LoggingCallbacks/types";

interface AccessControlFieldsProps {
  // value/onChange are optional so the component can be driven either directly
  // (the Add modal) or injected by an antd Form.Item (the Edit modal).
  value?: CredentialAccess;
  onChange?: (next: CredentialAccess) => void;
}

// Admin-owned access for a logging destination: global (every request) or a set of
// teams/orgs. Per-key targeting is intentionally absent here -- it lives on the key's
// own page, since a key's token rotates on regenerate while team/org ids are stable.
const AccessControlFields: React.FC<AccessControlFieldsProps> = ({ value = {}, onChange = () => {} }) => {
  const { data: teams } = useTeams();
  const { data: orgs } = useOrganizations();
  const isGlobal = value.global === true;

  const teamOptions = (teams ?? []).map((t) => ({ value: t.team_id, label: t.team_alias || t.team_id }));
  const orgOptions = (orgs ?? []).map((o) => ({
    value: o.organization_id,
    label: o.organization_alias || o.organization_id,
  }));

  return (
    <>
      <Form.Item
        label="Global"
        tooltip="When on, every request's traces export to this destination, regardless of key, team, or org."
      >
        <Switch checked={isGlobal} onChange={(global) => onChange({ ...value, global })} />
      </Form.Item>
      <Form.Item label="Teams" tooltip="Requests from keys in these teams export to this destination.">
        <Select
          mode="multiple"
          allowClear
          disabled={isGlobal}
          placeholder="Select teams"
          value={value.teams ?? []}
          onChange={(teamIds) => onChange({ ...value, teams: teamIds })}
          options={teamOptions}
          optionFilterProp="label"
          style={{ width: "100%" }}
        />
      </Form.Item>
      <Form.Item label="Organizations" tooltip="Requests under these orgs export to this destination.">
        <Select
          mode="multiple"
          allowClear
          disabled={isGlobal}
          placeholder="Select organizations"
          value={value.orgs ?? []}
          onChange={(orgIds) => onChange({ ...value, orgs: orgIds })}
          options={orgOptions}
          optionFilterProp="label"
          style={{ width: "100%" }}
        />
      </Form.Item>
    </>
  );
};

export default AccessControlFields;
