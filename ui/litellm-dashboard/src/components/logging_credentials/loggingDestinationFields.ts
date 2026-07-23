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
  // Example value shown as the input placeholder, so an admin knows the format.
  placeholder?: string;
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
      {
        name: "langfuse_host",
        label: "Langfuse Host",
        type: "text",
        placeholder: "https://cloud.langfuse.com",
      },
      {
        name: "langfuse_public_key",
        label: "Public Key",
        type: "password",
        placeholder: "pk-lf-00000000-0000-0000-0000-000000000000",
      },
      {
        name: "langfuse_secret_key",
        label: "Secret Key",
        type: "password",
        placeholder: "sk-lf-00000000-0000-0000-0000-000000000000",
      },
    ],
    hostField: "langfuse_host",
  },
  {
    id: "arize",
    label: "Arize",
    fields: [
      {
        name: "arize_space_id",
        label: "Space ID",
        type: "password",
        placeholder: "U3BhY2U6MTIzNDU6YWJjZA==",
      },
      {
        name: "arize_api_key",
        label: "API Key",
        type: "password",
        placeholder: "ak-0000aaaa-1111-2222-3333-444455556666",
      },
      {
        name: "arize_project_name",
        label: "Project Name",
        type: "text",
        placeholder: "my-llm-app",
      },
      {
        name: "arize_endpoint",
        label: "Endpoint",
        type: "text",
        optional: true,
        placeholder: "https://otlp.arize.com/v1",
      },
    ],
    hostField: "arize_endpoint",
  },
  {
    id: "weave_otel",
    label: "Weave",
    fields: [
      {
        name: "wandb_api_key",
        label: "W&B API Key",
        type: "password",
        placeholder: "0123456789abcdef0123456789abcdef01234567",
      },
      {
        name: "weave_project_id",
        label: "Project (entity/project)",
        type: "text",
        placeholder: "my-team/my-project",
      },
      {
        name: "weave_endpoint",
        label: "Endpoint",
        type: "text",
        optional: true,
        placeholder: "https://trace.wandb.ai",
      },
    ],
    hostField: "weave_endpoint",
  },
  {
    id: "generic",
    label: "Generic OTLP Collector",
    fields: [
      {
        name: "otel_endpoint",
        label: "OTLP Endpoint",
        type: "text",
        placeholder: "https://collector.example.com:4318/v1/traces",
      },
      {
        name: "otel_headers",
        label: "Headers (k=v,k2=v2)",
        type: "text",
        optional: true,
        placeholder: "x-api-key=abc123,x-team=42",
      },
    ],
    hostField: "otel_endpoint",
  },
];
