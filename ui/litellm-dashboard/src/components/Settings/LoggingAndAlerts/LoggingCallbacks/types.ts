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
  // The destination's whole stored credential_info. PATCH replaces
  // credential_info wholesale, so an access edit has to resend all of it.
  credentialInfo?: Record<string, unknown>;
  // The set of identities that route to this destination, resolved at render
  // time from credential_info.access. Display labels only -- ids are not
  // surfaced here. global=true bypasses the lists.
  resolvedScope?: ResolvedScope;
  // Whether the backend can actually build an exporter from this credential, decided
  // there by the same function the request-time resolver and the team/org disclosure
  // use. Read rather than recomputed: a second implementation of the adapter rules
  // here would drift, and the drift is what let a dead destination look active.
  resolvesToDestination?: boolean;
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
