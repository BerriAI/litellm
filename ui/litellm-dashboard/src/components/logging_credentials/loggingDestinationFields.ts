// Create-time field shapes for an admin-owned logging destination, keyed by the
// OTEL v2 backend it binds to. This is the inverse of the per-team picker: the
// picker selects a destination by name; these fields are what an admin types when
// CREATING the named destination in the registry. Keeping the raw keys here (the
// admin registry) and out of the per-team form is the provider/logging separation.

export type LoggingFieldType = "text" | "password";

export interface LoggingField {
  name: string;
  label: string;
  type: LoggingFieldType;
  optional?: boolean;
}

export interface LoggingDestinationBackend {
  id: string; // the callback_name the credential is bound under
  label: string;
  fields: LoggingField[];
  // The non-secret field that names the destination host/endpoint. Surfaced in the
  // list so an admin can tell e.g. an EU from a US destination apart.
  hostField: string;
}

export const LOGGING_DESTINATION_BACKENDS: LoggingDestinationBackend[] = [
  {
    id: "langfuse_otel",
    label: "Langfuse",
    fields: [
      { name: "langfuse_host", label: "Langfuse Host", type: "text" },
      { name: "langfuse_public_key", label: "Public Key", type: "password" },
      { name: "langfuse_secret_key", label: "Secret Key", type: "password" },
    ],
    hostField: "langfuse_host",
  },
  {
    id: "arize",
    label: "Arize",
    fields: [
      { name: "arize_space_id", label: "Space ID", type: "password" },
      { name: "arize_api_key", label: "API Key", type: "password" },
      { name: "arize_endpoint", label: "Endpoint (optional)", type: "text", optional: true },
    ],
    hostField: "arize_endpoint",
  },
  {
    id: "weave_otel",
    label: "Weave",
    fields: [
      { name: "wandb_api_key", label: "W&B API Key", type: "password" },
      { name: "weave_endpoint", label: "Weave OTEL Endpoint", type: "text" },
      { name: "weave_project_id", label: "Project (entity/project)", type: "text" },
    ],
    hostField: "weave_endpoint",
  },
  {
    id: "generic",
    label: "Generic OTLP Collector",
    fields: [
      { name: "otel_endpoint", label: "OTLP Endpoint", type: "text" },
      { name: "otel_headers", label: "Headers (k=v,k2=v2)", type: "text", optional: true },
    ],
    hostField: "otel_endpoint",
  },
];
