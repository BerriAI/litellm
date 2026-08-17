import React from "react";

import { useOrganizations } from "@/app/(dashboard)/hooks/organizations/useOrganizations";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { Field, FieldDescription, FieldLabel } from "@/components/shared/form/field";
import { MultiSelect } from "@/components/shared/MultiSelect";
import { Switch } from "@/components/ui/switch";
import { CredentialAccess } from "../Settings/LoggingAndAlerts/LoggingCallbacks/types";

interface AccessControlFieldsProps {
  value?: CredentialAccess;
  onChange?: (next: CredentialAccess) => void;
}

// Admin-owned access for a logging destination: global (every request) or a set of
// teams/orgs. Per-key targeting is intentionally absent here -- it lives on the key's
// own page, since a key's token rotates on regenerate while team/org ids are stable.
const AccessControlFields: React.FC<AccessControlFieldsProps> = ({ value = {}, onChange = () => {} }) => {
  const { data: teams } = useTeams();
  const { data: orgs } = useOrganizations();
  const globalSwitchId = React.useId();
  const isGlobal = value.global === true;

  const teamOptions = (teams ?? []).map((t) => ({ value: t.team_id, label: t.team_alias || t.team_id }));
  const orgOptions = (orgs ?? []).map((o) => ({
    value: o.organization_id,
    label: o.organization_alias || o.organization_id,
  }));

  return (
    <>
      <Field orientation="horizontal">
        <FieldLabel htmlFor={globalSwitchId}>Global</FieldLabel>
        <Switch id={globalSwitchId} checked={isGlobal} onCheckedChange={(global) => onChange({ ...value, global })} />
        <FieldDescription>Traces from every team and org export to this destination</FieldDescription>
      </Field>
      <Field>
        <FieldLabel>Teams</FieldLabel>
        <MultiSelect
          options={teamOptions}
          value={value.teams ?? []}
          onValueChange={(teamIds) => onChange({ ...value, teams: teamIds })}
          placeholder="Select teams"
          disabled={isGlobal}
        />
        <FieldDescription>Only these teams&apos; traffic exports to this destination</FieldDescription>
      </Field>
      <Field>
        <FieldLabel>Organizations</FieldLabel>
        <MultiSelect
          options={orgOptions}
          value={value.orgs ?? []}
          onValueChange={(orgIds) => onChange({ ...value, orgs: orgIds })}
          placeholder="Select organizations"
          disabled={isGlobal}
        />
        <FieldDescription>Only these orgs&apos; traffic exports to this destination</FieldDescription>
      </Field>
    </>
  );
};

export default AccessControlFields;
