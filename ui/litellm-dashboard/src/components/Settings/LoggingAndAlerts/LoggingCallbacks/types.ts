export interface AlertingObject {
  name: string;
  // Backend distinguishes success vs failure callback registrations
  // (`/get_callbacks` returns `type: "success" | "failure"`). Same callback
  // (e.g. `generic_api`) can appear twice — once per event class — and
  // those entries fire on disjoint events, not double-fire on one event.
  // UI must read this to render the correct badge; missing it caused
  // every row to render as "Success".
  type?: "success" | "failure" | "success_and_failure";
  variables: AlertingVariables;
  // Present only on rows backed by a logging credential (an OTEL trace
  // destination). Config-callback rows leave these unset, which is how the table
  // tells the two apart.
  credentialName?: string;
  destinationLabel?: string;
  access?: CredentialAccess;
  // True when credential_info.auto_enable=true: destination exports on every
  // request without needing explicit key/team/org assignment. Distinct from
  // access.global (which controls visibility/assignability, not routing).
  autoEnable?: boolean;
  // The union of identities that route to this destination, resolved at render
  // time from both directions (destination-side credential_info.access AND
  // identity-side metadata.logging_exporters). Display labels only -- ids are
  // not surfaced here. global=true bypasses the lists.
  resolvedScope?: ResolvedScope;
}

export interface CredentialAccess {
  global?: boolean;
  teams?: string[];
  orgs?: string[];
}

export interface ResolvedScope {
  global: boolean;
  teams: string[];
  orgs: string[];
}

export interface AlertingVariables {
  SLACK_WEBHOOK_URL: string | null;
  LANGFUSE_PUBLIC_KEY: string | null;
  LANGFUSE_SECRET_KEY: string | null;
  LANGFUSE_HOST: string | null;
  OPENMETER_API_KEY: string | null;
}
