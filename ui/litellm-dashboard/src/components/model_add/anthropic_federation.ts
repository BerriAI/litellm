import type { ParseKeys, TFunction } from "i18next";
import i18n from "@/i18n/i18n";
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
  readonly labelKey: ParseKeys;
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
    labelKey: "modelAdd.anthropicFederation.fields.federationRuleId.label",
    tooltipKey: "modelAdd.anthropicFederation.fields.federationRuleId.tooltip",
    placeholder: "fdrl_...",
    required: false,
    control: "text",
  },
  {
    key: "anthropic_organization_id",
    labelKey: "modelAdd.anthropicFederation.fields.organizationId.label",
    tooltipKey: "modelAdd.anthropicFederation.fields.organizationId.tooltip",
    required: false,
    control: "text",
  },
  {
    key: "anthropic_service_account_id",
    labelKey: "modelAdd.anthropicFederation.fields.serviceAccountId.label",
    tooltipKey: "modelAdd.anthropicFederation.fields.serviceAccountId.tooltip",
    placeholder: "svac_...",
    required: false,
    control: "text",
  },
  {
    key: "anthropic_federation_workspace_id",
    labelKey: "modelAdd.anthropicFederation.fields.workspaceId.label",
    tooltipKey: "modelAdd.anthropicFederation.fields.workspaceId.tooltip",
    placeholder: "wrkspc_...",
    required: false,
    control: "text",
  },
];

export const IDENTITY_SOURCES: readonly IdentitySource[] = [
  {
    id: "token_file",
    labelKey: "modelAdd.anthropicFederation.sources.tokenFile",
    fixedValues: {},
    fields: [
      {
        key: "anthropic_identity_token_file",
        labelKey: "modelAdd.anthropicFederation.fields.identityTokenFile.label",
        tooltipKey: "modelAdd.anthropicFederation.fields.identityTokenFile.tooltip",
        placeholder: "/var/run/secrets/anthropic/token",
        required: true,
        control: "text",
      },
    ],
  },
  {
    id: "secret_reference",
    labelKey: "modelAdd.anthropicFederation.sources.secretReference",
    fixedValues: {},
    fields: [
      {
        key: "anthropic_identity_token",
        labelKey: "modelAdd.anthropicFederation.fields.identityTokenReference.label",
        tooltipKey: "modelAdd.anthropicFederation.fields.identityTokenReference.tooltip",
        placeholder: "oidc/env/ANTHROPIC_IDENTITY_TOKEN",
        required: true,
        control: "text",
      },
    ],
  },
  {
    id: "internal_issuer",
    labelKey: "modelAdd.anthropicFederation.sources.internalIssuer",
    fixedValues: { [IDENTITY_SOURCE_KEY]: "internal_issuer" },
    fields: [
      {
        key: "anthropic_issuer_url",
        labelKey: "modelAdd.anthropicFederation.fields.issuerUrl.label",
        tooltipKey: "modelAdd.anthropicFederation.fields.issuerUrl.tooltip",
        placeholder: "https://litellm.example.com",
        required: true,
        control: "text",
      },
      {
        key: "anthropic_issuer_subject",
        labelKey: "modelAdd.anthropicFederation.fields.issuerSubject.label",
        tooltipKey: "modelAdd.anthropicFederation.fields.issuerSubject.tooltip",
        required: true,
        control: "text",
      },
      {
        key: "anthropic_issuer_audience",
        labelKey: "modelAdd.anthropicFederation.fields.issuerAudience.label",
        tooltipKey: "modelAdd.anthropicFederation.fields.issuerAudience.tooltip",
        required: false,
        control: "text",
      },
      {
        key: "anthropic_issuer_ttl_seconds",
        labelKey: "modelAdd.anthropicFederation.fields.issuerTtlSeconds.label",
        tooltipKey: "modelAdd.anthropicFederation.fields.issuerTtlSeconds.tooltip",
        tooltipValues: { max: MAX_ISSUER_TTL_SECONDS },
        required: false,
        control: "integer",
      },
      {
        key: "anthropic_issuer_signing_key_ref",
        labelKey: "modelAdd.anthropicFederation.fields.issuerSigningKeyRef.label",
        tooltipKey: "modelAdd.anthropicFederation.fields.issuerSigningKeyRef.tooltip",
        placeholder: "os.environ/ANTHROPIC_ISSUER_SIGNING_KEY",
        required: true,
        control: "text",
      },
    ],
  },
  {
    id: "keycloak",
    labelKey: "modelAdd.anthropicFederation.sources.keycloak",
    fixedValues: { [IDENTITY_SOURCE_KEY]: "keycloak" },
    fields: [
      {
        key: "anthropic_keycloak_token_url",
        labelKey: "modelAdd.anthropicFederation.fields.keycloakTokenUrl.label",
        tooltipKey: "modelAdd.anthropicFederation.fields.keycloakTokenUrl.tooltip",
        placeholder: "https://keycloak.example.com/realms/<realm>/protocol/openid-connect/token",
        required: true,
        control: "text",
      },
      {
        key: "anthropic_keycloak_client_id",
        labelKey: "modelAdd.anthropicFederation.fields.keycloakClientId.label",
        tooltipKey: "modelAdd.anthropicFederation.fields.keycloakClientId.tooltip",
        required: true,
        control: "text",
      },
      {
        key: "anthropic_keycloak_client_secret_ref",
        labelKey: "modelAdd.anthropicFederation.fields.keycloakClientSecretRef.label",
        tooltipKey: "modelAdd.anthropicFederation.fields.keycloakClientSecretRef.tooltip",
        placeholder: "os.environ/KEYCLOAK_CLIENT_SECRET",
        required: true,
        control: "text",
      },
      {
        key: "anthropic_keycloak_auth_method",
        labelKey: "modelAdd.anthropicFederation.fields.keycloakAuthMethod.label",
        tooltipKey: "modelAdd.anthropicFederation.fields.keycloakAuthMethod.tooltip",
        required: false,
        control: "select",
        options: ["client_secret_basic", "client_secret_post"],
      },
      {
        key: "anthropic_keycloak_scope",
        labelKey: "modelAdd.anthropicFederation.fields.keycloakScope.label",
        tooltipKey: "modelAdd.anthropicFederation.fields.keycloakScope.tooltip",
        required: false,
        control: "text",
      },
    ],
  },
  {
    id: "environment",
    labelKey: "modelAdd.anthropicFederation.sources.environment",
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

const UNRECOGNIZED_IDENTITY_SOURCE: IdentitySource = {
  id: "unrecognized",
  labelKey: "modelAdd.anthropicFederation.storedSource",
  fixedValues: {},
  fields: [],
};

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
  t: TFunction = i18n.t.bind(i18n),
): readonly { value: IdentitySourceId; label: string }[] => [
  ...(inferIdentitySource(storedValues) === "unrecognized"
    ? [
        {
          value: "unrecognized" as const,
          label: t("modelAdd.anthropicFederation.storedSource", { source: String(storedValues[IDENTITY_SOURCE_KEY]) }),
        },
      ]
    : []),
  ...IDENTITY_SOURCES.map((source) => ({ value: source.id, label: t(source.labelKey) })),
];

export const validateFederationValueStored =
  (identitySource: IdentitySourceId) =>
  (_value: unknown, formValues: Record<string, unknown>): string | true =>
    identitySource === "environment" && FEDERATION_CORE_FIELDS.every((field) => isBlank(formValues[field.key]))
      ? i18n.t("modelAdd.anthropicFederation.storedValueRequired")
      : true;

export const validateIdentityTokenReference = (value: unknown): string | true => {
  if (typeof value !== "string" || isBlank(value) || isMaskedSecret(value)) {
    return true;
  }
  return value.startsWith(ACCEPTED_REFERENCE_PREFIX) && !value.startsWith(REJECTED_REFERENCE_PREFIX)
    ? true
    : i18n.t("modelAdd.anthropicFederation.invalidTokenReference");
};

export const validateIssuerTtlSeconds = (value: unknown): string | true => {
  if (isBlank(value)) {
    return true;
  }
  const seconds = Number(value);
  return Number.isInteger(seconds) && seconds >= 1 && seconds <= MAX_ISSUER_TTL_SECONDS
    ? true
    : i18n.t("modelAdd.anthropicFederation.invalidTtl", { max: MAX_ISSUER_TTL_SECONDS });
};

export const otherIdentitySourceKeys = (identitySource: IdentitySourceId): readonly string[] => {
  const source = identitySourceById(identitySource);
  const kept = new Set([...source.fields.map((field) => field.key), ...Object.keys(source.fixedValues)]);
  return IDENTITY_SOURCE_VALUE_KEYS.filter((key) => !kept.has(key));
};
