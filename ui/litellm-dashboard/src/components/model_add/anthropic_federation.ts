import { isMaskedSecret } from "@/utils/maskedSecretUtils";
import { isBlank, type FederationField } from "./federation_field";

export type IdentitySourceId =
  | "token_file"
  | "secret_reference"
  | "internal_issuer"
  | "keycloak"
  | "environment"
  | "unrecognized";

export interface IdentitySource {
  readonly id: IdentitySourceId;
  readonly label: string;
  readonly fixedValues: Readonly<Record<string, string>>;
  readonly fields: readonly FederationField[];
}

export const MAX_ISSUER_TTL_SECONDS = 3600;

const IDENTITY_SOURCE_KEY = "anthropic_identity_source";
const ACCEPTED_REFERENCE_PREFIX = "oidc/";
const REJECTED_REFERENCE_PREFIX = "oidc/env_path/";

export const FEDERATION_CORE_FIELDS: readonly FederationField[] = [
  {
    key: "anthropic_federation_rule_id",
    label: "Federation Rule ID",
    tooltip:
      "The fdrl_ id of the federation rule, from Settings > Workload identity in the Claude Console. Leave empty when the proxy sets ANTHROPIC_FEDERATION_RULE_ID in its environment.",
    placeholder: "fdrl_...",
    required: false,
    control: "text",
  },
  {
    key: "anthropic_organization_id",
    label: "Organization ID",
    tooltip:
      "The Anthropic organization the federation rule belongs to, shown on the rule's detail page. Leave empty when the proxy sets ANTHROPIC_ORGANIZATION_ID in its environment.",
    required: false,
    control: "text",
  },
  {
    key: "anthropic_service_account_id",
    label: "Service Account ID",
    tooltip: "The svac_ id the federation rule targets. Anthropic's reference lists it as required.",
    placeholder: "svac_...",
    required: false,
    control: "text",
  },
  {
    key: "anthropic_federation_workspace_id",
    label: "Workspace ID",
    tooltip:
      "Set this when the federation rule is enabled in more than one workspace: the wrkspc_ id to mint tokens for, or 'default'.",
    placeholder: "wrkspc_...",
    required: false,
    control: "text",
  },
];

export const IDENTITY_SOURCES: readonly IdentitySource[] = [
  {
    id: "token_file",
    label: "Identity token file",
    fixedValues: {},
    fields: [
      {
        key: "anthropic_identity_token_file",
        label: "Identity Token File",
        tooltip:
          "Absolute path on the proxy host of a file holding the identity token, for example a projected Kubernetes service account token. It must sit under an allowed credential directory.",
        placeholder: "/var/run/secrets/anthropic/token",
        required: true,
        control: "text",
      },
    ],
  },
  {
    id: "secret_reference",
    label: "Identity token secret reference",
    fixedValues: {},
    fields: [
      {
        key: "anthropic_identity_token",
        label: "Identity Token Reference",
        tooltip:
          "An oidc/ secret reference the proxy resolves on each exchange, such as oidc/env/VAR_NAME, oidc/github/<audience>, or oidc/google/<audience>. Raw tokens are not accepted.",
        placeholder: "oidc/env/ANTHROPIC_IDENTITY_TOKEN",
        required: true,
        control: "text",
      },
    ],
  },
  {
    id: "internal_issuer",
    label: "Token signed by LiteLLM (internal issuer)",
    fixedValues: { [IDENTITY_SOURCE_KEY]: "internal_issuer" },
    fields: [
      {
        key: "anthropic_issuer_url",
        label: "Issuer URL",
        tooltip: "The issuer registered on the federation rule. The proxy puts it in the token's iss claim.",
        placeholder: "https://litellm.example.com",
        required: true,
        control: "text",
      },
      {
        key: "anthropic_issuer_subject",
        label: "Subject",
        tooltip: "The sub claim the federation rule matches.",
        required: true,
        control: "text",
      },
      {
        key: "anthropic_issuer_audience",
        label: "Audience",
        tooltip: "Optional aud claim, when the federation rule checks one.",
        required: false,
        control: "text",
      },
      {
        key: "anthropic_issuer_ttl_seconds",
        label: "Token Lifetime (seconds)",
        tooltip: `Optional. How long each signed token is valid, from 1 to ${MAX_ISSUER_TTL_SECONDS} seconds. The proxy uses its default when this is empty.`,
        required: false,
        control: "integer",
      },
      {
        key: "anthropic_issuer_signing_key_ref",
        label: "Signing Key Reference",
        tooltip:
          "A secret reference to the PEM private key the proxy signs with, such as os.environ/VAR_NAME. Enter the reference, never the key itself.",
        placeholder: "os.environ/ANTHROPIC_ISSUER_SIGNING_KEY",
        required: true,
        control: "text",
      },
    ],
  },
  {
    id: "keycloak",
    label: "Keycloak client credentials",
    fixedValues: { [IDENTITY_SOURCE_KEY]: "keycloak" },
    fields: [
      {
        key: "anthropic_keycloak_token_url",
        label: "Keycloak Token URL",
        tooltip: "The realm's token endpoint the proxy requests an identity token from.",
        placeholder: "https://keycloak.example.com/realms/<realm>/protocol/openid-connect/token",
        required: true,
        control: "text",
      },
      {
        key: "anthropic_keycloak_client_id",
        label: "Keycloak Client ID",
        tooltip: "The client the proxy authenticates as.",
        required: true,
        control: "text",
      },
      {
        key: "anthropic_keycloak_client_secret_ref",
        label: "Client Secret Reference",
        tooltip:
          "A secret reference to the client secret, such as os.environ/VAR_NAME. Enter the reference, never the secret itself.",
        placeholder: "os.environ/KEYCLOAK_CLIENT_SECRET",
        required: true,
        control: "text",
      },
      {
        key: "anthropic_keycloak_auth_method",
        label: "Client Authentication Method",
        tooltip: "Optional. How the client secret is sent to Keycloak. The proxy uses client_secret_basic when unset.",
        required: false,
        control: "select",
        options: ["client_secret_basic", "client_secret_post"],
      },
      {
        key: "anthropic_keycloak_scope",
        label: "Scope",
        tooltip: "Optional scope to request with the token.",
        required: false,
        control: "text",
      },
    ],
  },
  {
    id: "environment",
    label: "Proxy environment variables",
    fixedValues: {},
    fields: [],
  },
];

export const DEFAULT_IDENTITY_SOURCE: IdentitySourceId = "token_file";

const IDENTITY_SOURCE_VALUE_KEYS: readonly string[] = [
  IDENTITY_SOURCE_KEY,
  ...IDENTITY_SOURCES.flatMap((source) => source.fields.map((field) => field.key)),
];

export const ANTHROPIC_FEDERATION_FIELDS: readonly FederationField[] = [
  ...FEDERATION_CORE_FIELDS,
  ...IDENTITY_SOURCES.flatMap((source) => source.fields),
];

export const ANTHROPIC_FEDERATION_VALUE_KEYS: readonly string[] = [
  ...FEDERATION_CORE_FIELDS.map((field) => field.key),
  ...IDENTITY_SOURCE_VALUE_KEYS,
];

const UNRECOGNIZED_IDENTITY_SOURCE: IdentitySource = { id: "unrecognized", label: "", fixedValues: {}, fields: [] };

export const identitySourceById = (id: IdentitySourceId): IdentitySource =>
  IDENTITY_SOURCES.find((source) => source.id === id) ?? UNRECOGNIZED_IDENTITY_SOURCE;

export const isAnthropicProvider = (provider: string | null | undefined): boolean =>
  provider !== null && provider !== undefined && provider.toLowerCase() === "anthropic";

export const inferIdentitySource = (credentialValues: Record<string, unknown> | null | undefined): IdentitySourceId => {
  const values = credentialValues ?? {};
  const declared = values[IDENTITY_SOURCE_KEY];
  if (!isBlank(declared)) {
    return (
      IDENTITY_SOURCES.find((source) => source.fixedValues[IDENTITY_SOURCE_KEY] === declared)?.id ?? "unrecognized"
    );
  }
  if (!isBlank(values.anthropic_identity_token_file)) {
    return "token_file";
  }
  if (!isBlank(values.anthropic_identity_token)) {
    return "secret_reference";
  }
  return "environment";
};

export const identitySourceOptions = (
  storedValues: Record<string, unknown>,
): readonly { value: IdentitySourceId; label: string }[] => [
  ...(inferIdentitySource(storedValues) === "unrecognized"
    ? [{ value: "unrecognized" as const, label: `Stored: ${String(storedValues[IDENTITY_SOURCE_KEY])}` }]
    : []),
  ...IDENTITY_SOURCES.map((source) => ({ value: source.id, label: source.label })),
];

export const validateFederationValueStored =
  (identitySource: IdentitySourceId) =>
  (_value: unknown, formValues: Record<string, unknown>): string | true =>
    identitySource === "environment" && FEDERATION_CORE_FIELDS.every((field) => isBlank(formValues[field.key]))
      ? "Enter at least one of these ids, or pick an identity source that stores a token. The proxy rejects a credential with no values"
      : true;

export const validateIdentityTokenReference = (value: unknown): string | true => {
  if (typeof value !== "string" || isBlank(value) || isMaskedSecret(value)) {
    return true;
  }
  return value.startsWith(ACCEPTED_REFERENCE_PREFIX) && !value.startsWith(REJECTED_REFERENCE_PREFIX)
    ? true
    : "Enter an oidc/ secret reference such as oidc/env/VAR_NAME. Raw tokens and oidc/env_path/ references are not accepted";
};

export const validateIssuerTtlSeconds = (value: unknown): string | true => {
  if (isBlank(value)) {
    return true;
  }
  const seconds = Number(value);
  return Number.isInteger(seconds) && seconds >= 1 && seconds <= MAX_ISSUER_TTL_SECONDS
    ? true
    : `Enter a whole number of seconds from 1 to ${MAX_ISSUER_TTL_SECONDS}`;
};

export const otherIdentitySourceKeys = (identitySource: IdentitySourceId): readonly string[] => {
  const source = identitySourceById(identitySource);
  const kept = new Set([...source.fields.map((field) => field.key), ...Object.keys(source.fixedValues)]);
  return IDENTITY_SOURCE_VALUE_KEYS.filter((key) => !kept.has(key));
};
