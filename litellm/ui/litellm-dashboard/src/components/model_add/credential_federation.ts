import { isMaskedSecret } from "@/utils/maskedSecretUtils";
import type { CredentialItem } from "../networking";
import {
  ANTHROPIC_FEDERATION_FIELDS,
  ANTHROPIC_FEDERATION_VALUE_KEYS,
  identitySourceById,
  inferIdentitySource,
  isAnthropicProvider,
  otherIdentitySourceKeys,
  type IdentitySourceId,
} from "./anthropic_federation";
import { isBlank, type FederationField } from "./federation_field";
import { isOpenAIProvider, OPENAI_FEDERATION_FIELDS, validateOpenAIFederationApiBase } from "./openai_federation";

export type AuthMethod = "api_key" | "federation";

export type FederatedProvider = "anthropic" | "openai";

export type FederatedSelection =
  | { readonly authMethod: "federation"; readonly provider: "openai" }
  | { readonly authMethod: "federation"; readonly provider: "anthropic"; readonly identitySource: IdentitySourceId };

export type CredentialSelection = { readonly authMethod: "api_key" } | FederatedSelection;

export type ProviderFieldValidators = Readonly<Record<string, (value: unknown) => string | true>>;

export interface CredentialValuesPatch {
  readonly credential_values: Record<string, unknown>;
  readonly credential_values_to_delete: readonly string[];
}

const API_KEY = "api_key";

const FEDERATION_VALUE_KEYS: readonly string[] = [
  ...ANTHROPIC_FEDERATION_VALUE_KEYS,
  ...OPENAI_FEDERATION_FIELDS.map((field) => field.key),
];

const federationFieldsByKey: ReadonlyMap<string, FederationField> = new Map(
  [...ANTHROPIC_FEDERATION_FIELDS, ...OPENAI_FEDERATION_FIELDS].map((field) => [field.key, field]),
);

const OPENAI_FEDERATION_PROVIDER_FIELD_VALIDATORS: ProviderFieldValidators = {
  api_base: validateOpenAIFederationApiBase,
};

const NO_PROVIDER_FIELD_VALIDATORS: ProviderFieldValidators = {};

export const federatedProviderOf = (provider: string | null | undefined): FederatedProvider | null => {
  if (isAnthropicProvider(provider)) {
    return "anthropic";
  }
  return isOpenAIProvider(provider) ? "openai" : null;
};

export const selectionFor = (
  provider: FederatedProvider | null,
  authMethod: AuthMethod,
  identitySource: IdentitySourceId,
): CredentialSelection => {
  if (provider === null || authMethod === "api_key") {
    return { authMethod: "api_key" };
  }
  return provider === "anthropic"
    ? { authMethod: "federation", provider, identitySource }
    : { authMethod: "federation", provider };
};

export const isFederatedCredential = (credentialValues: Record<string, unknown> | null | undefined): boolean =>
  FEDERATION_VALUE_KEYS.some((key) => !isBlank(credentialValues?.[key]));

export const inferAuthMethod = (credentialValues: Record<string, unknown> | null | undefined): AuthMethod =>
  isBlank(credentialValues?.[API_KEY]) && isFederatedCredential(credentialValues) ? "federation" : "api_key";

export const providerFieldValidators = (selection: CredentialSelection): ProviderFieldValidators =>
  selection.authMethod === "federation" && selection.provider === "openai"
    ? OPENAI_FEDERATION_PROVIDER_FIELD_VALIDATORS
    : NO_PROVIDER_FIELD_VALIDATORS;

const identitySourceOf = (selection: CredentialSelection): IdentitySourceId | null =>
  selection.authMethod === "federation" && selection.provider === "anthropic" ? selection.identitySource : null;

export type JwksPanel =
  | { readonly kind: "hidden" }
  | { readonly kind: "after_save" }
  | { readonly kind: "saved"; readonly credentialName: string };

export const jwksPanelFor = (storedCredential: CredentialItem | null, selection: CredentialSelection): JwksPanel => {
  if (identitySourceOf(selection) !== "internal_issuer") {
    return { kind: "hidden" };
  }
  if (
    storedCredential === null ||
    !isAnthropicProvider(storedCredential.credential_info.custom_llm_provider) ||
    inferIdentitySource(storedCredential.credential_values) !== "internal_issuer"
  ) {
    return { kind: "after_save" };
  }
  return { kind: "saved", credentialName: storedCredential.credential_name };
};

const toStoredType = (field: FederationField | undefined, value: unknown): unknown => {
  if (field === undefined || isBlank(value)) {
    return field === undefined ? value : "";
  }
  if (field.control === "integer") {
    return Number(value);
  }
  return typeof value === "string" ? value.trim() : value;
};

const fixedValuesFor = (selection: CredentialSelection): Readonly<Record<string, string>> => {
  const identitySource = identitySourceOf(selection);
  return identitySource === null ? {} : identitySourceById(identitySource).fixedValues;
};

const typedFormValues = (formValues: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(
    Object.entries(formValues)
      .map(([key, value]) => [key, toStoredType(federationFieldsByKey.get(key), value)] as const)
      .filter(([, value]) => value !== "" && value !== undefined && value !== null),
  );

export const buildCreateCredentialValues = (
  formValues: Record<string, unknown>,
  selection: CredentialSelection,
): Record<string, unknown> => ({ ...typedFormValues(formValues), ...fixedValuesFor(selection) });

const keysLeftBehind = (initial: CredentialSelection, selection: CredentialSelection): readonly string[] => {
  if (selection.authMethod !== initial.authMethod) {
    return selection.authMethod === "federation" ? [API_KEY] : FEDERATION_VALUE_KEYS;
  }
  const identitySource = identitySourceOf(selection);
  return identitySource === null || identitySource === identitySourceOf(initial)
    ? []
    : otherIdentitySourceKeys(identitySource);
};

const changedValues = (stored: Record<string, unknown>, desired: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(Object.entries(desired).filter(([key, value]) => !isMaskedSecret(value) && value !== stored[key]));

export const buildProviderChangePatch = (
  stored: Record<string, unknown>,
  formValues: Record<string, unknown>,
  selection: CredentialSelection,
): CredentialValuesPatch => {
  const desired = { ...typedFormValues(formValues), ...fixedValuesFor(selection) };
  return {
    credential_values: changedValues(stored, desired),
    credential_values_to_delete: Object.keys(stored).filter((key) => !(key in desired)),
  };
};

export const buildCredentialPatch = (
  stored: Record<string, unknown>,
  formValues: Record<string, unknown>,
  initial: CredentialSelection,
  selection: CredentialSelection,
): CredentialValuesPatch => {
  const changed = changedValues(stored, { ...typedFormValues(formValues), ...fixedValuesFor(selection) });
  const cleared = Object.keys(formValues).filter((key) => isBlank(formValues[key]) && !isBlank(stored[key]));
  const toDelete = [...keysLeftBehind(initial, selection), ...cleared].filter(
    (key) => key in stored && !(key in changed),
  );
  return { credential_values: changed, credential_values_to_delete: Array.from(new Set(toDelete)) };
};
