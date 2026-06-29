import { CredentialAccess } from "../Settings/LoggingAndAlerts/LoggingCallbacks/types";
import { credentialCreateCall } from "../networking";
import { LOGGING_DESTINATION_BACKENDS } from "./loggingDestinationFields";

// The set of OTEL backend ids that are created as logging destinations (credentials),
// not as global config callbacks. The unified Add modal branches on this.
export const LOGGING_BACKEND_IDS: ReadonlySet<string> = new Set(LOGGING_DESTINATION_BACKENDS.map((b) => b.id));

// Callback ids that must not surface as global callbacks. Per LIT-3850 OTEL is admin-
// owned and routed per identity via trace destinations, and the legacy Langfuse/OTEL
// callback paths (`langfuse` v2 SDK, `langfuse_otel` v1, the generic `otel` callback)
// are deprecated, so these are only ever destinations -- never callback rows or options.
export const NON_CALLBACK_LOGGING_IDS: ReadonlySet<string> = new Set([
  ...LOGGING_DESTINATION_BACKENDS.map((b) => b.id),
  "langfuse",
  "otel",
]);

export const backendLabel = (id?: string): string =>
  LOGGING_DESTINATION_BACKENDS.find((b) => b.id === id)?.label ?? id ?? "-";

export interface CreateLoggingCredentialInput {
  credentialName: string;
  backend: string;
  values: Record<string, string>;
  host?: string;
  access?: CredentialAccess;
  autoEnable?: boolean;
}

// One place that owns the logging-credential contract: the credential_type tag, the
// backend in description, the non-secret host, the admin-owned access grant, and the
// explicit global/default (auto_enable) opt-in.
export const createLoggingCredential = async (accessToken: string, input: CreateLoggingCredentialInput) =>
  credentialCreateCall(accessToken, {
    credential_name: input.credentialName,
    credential_values: input.values,
    credential_info: {
      credential_type: "logging",
      description: input.backend,
      ...(input.host ? { host: input.host } : {}),
      ...(input.access ? { access: input.access } : {}),
      ...(input.autoEnable ? { auto_enable: true } : {}),
    },
  });
