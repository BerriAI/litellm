import { CredentialAccess } from "../Settings/LoggingAndAlerts/LoggingCallbacks/types";
import { credentialCreateCall } from "../networking";
import { LOGGING_DESTINATION_BACKENDS } from "./loggingDestinationFields";

// The OTEL trace-destination backend ids, managed as admin-owned logging destinations
// (credentials). The `langfuse` v2 SDK logger and the generic `otel` callback are
// deliberately not here; they keep their existing global-callback behavior untouched.
export const LOGGING_BACKEND_IDS: ReadonlySet<string> = new Set(LOGGING_DESTINATION_BACKENDS.map((b) => b.id));

// These backends are reachable two ways from the Add modal, so the dropdown carries a
// prefixed option id for the destination branch. Without it a bare `arize` would be
// ambiguous, and picking it would silently create an inert credential instead of the
// proxy-wide callback the same label used to produce.
export const DESTINATION_OPTION_PREFIX = "destination:";

export const backendLabel = (id?: string): string =>
  LOGGING_DESTINATION_BACKENDS.find((b) => b.id === id)?.label ?? id ?? "-";

export interface CreateLoggingCredentialInput {
  credentialName: string;
  backend: string;
  values: Record<string, string>;
  host?: string;
  access?: CredentialAccess;
}

// One place that owns the logging-credential contract: the credential_type tag, the
// backend in description, the non-secret host, and the admin-owned access grant.
export const createLoggingCredential = async (accessToken: string, input: CreateLoggingCredentialInput) =>
  credentialCreateCall(accessToken, {
    credential_name: input.credentialName,
    credential_values: input.values,
    credential_info: {
      credential_type: "logging",
      description: input.backend,
      ...(input.host ? { host: input.host } : {}),
      ...(input.access ? { access: input.access } : {}),
    },
  });
