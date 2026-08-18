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
}

export interface AlertingVariables {
  SLACK_WEBHOOK_URL: string | null;
  LANGFUSE_PUBLIC_KEY: string | null;
  LANGFUSE_SECRET_KEY: string | null;
  LANGFUSE_HOST: string | null;
  OPENMETER_API_KEY: string | null;
}
