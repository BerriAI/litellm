import { Providers } from "../provider_info_helpers";

export interface ProviderAuthType {
  readonly id: string;
  readonly label: string;
  readonly description: string;
  readonly fieldKeys: readonly string[];
  readonly requiredFieldKeys: readonly string[];
}

const EMPTY_PROVIDER_AUTH_TYPES: readonly ProviderAuthType[] = [];

export const PROVIDER_AUTH_TYPES: Partial<Record<keyof typeof Providers, readonly ProviderAuthType[]>> = {
  MICROSOFT_365_COPILOT: [
    {
      id: "oauth_token_exchange",
      label: "OAuth token exchange (on-behalf-of)",
      description:
        "LiteLLM exchanges each caller's IdP-issued JWT at your IdP's token endpoint for a delegated token. Microsoft Graph only accepts Microsoft Entra tokens, so for Microsoft 365 Copilot the token endpoint must be Entra and callers must send an Entra-issued JWT for this app. Users can still sign in through any IdP federated with Entra.",
      fieldKeys: [
        "token_exchange_endpoint",
        "token_exchange_profile",
        "client_id",
        "client_secret",
        "token_exchange_scope",
        "token_exchange_audience",
      ],
      requiredFieldKeys: ["token_exchange_endpoint", "client_id", "client_secret"],
    },
    {
      id: "static_token",
      label: "Static delegated access token",
      description: "Sends one pre-acquired delegated Microsoft Graph token for every caller.",
      fieldKeys: ["api_key"],
      requiredFieldKeys: ["api_key"],
    },
  ],
};

export const authTypesFor = (provider: string | null): readonly ProviderAuthType[] => {
  const providerKeys: readonly (keyof typeof Providers)[] = Object.keys(Providers) as (keyof typeof Providers)[];
  const providerKey =
    provider === null ? undefined : providerKeys.find((key) => key === provider || Providers[key] === provider);

  return providerKey === undefined
    ? EMPTY_PROVIDER_AUTH_TYPES
    : PROVIDER_AUTH_TYPES[providerKey] ?? EMPTY_PROVIDER_AUTH_TYPES;
};

export const authTypeFieldKeys = (provider: string | null): readonly string[] =>
  authTypesFor(provider)
    .flatMap(({ fieldKeys }) => fieldKeys)
    .filter((fieldKey, index, fieldKeys) => fieldKeys.indexOf(fieldKey) === index);

export const inferAuthTypeId = (authTypes: readonly ProviderAuthType[], values: Record<string, unknown>): string =>
  authTypes.find(({ requiredFieldKeys }) =>
    requiredFieldKeys.some(
      (fieldKey) => values[fieldKey] !== undefined && values[fieldKey] !== null && values[fieldKey] !== "",
    ),
  )?.id ??
  authTypes[0]?.id ??
  "";

export const hiddenAuthFieldKeys = (authTypes: readonly ProviderAuthType[], selectedId: string): readonly string[] => {
  const selectedFieldKeys = authTypes.find(({ id }) => id === selectedId)?.fieldKeys ?? [];

  return authTypes
    .filter(({ id }) => id !== selectedId)
    .flatMap(({ fieldKeys }) => fieldKeys)
    .filter(
      (fieldKey, index, fieldKeys) => !selectedFieldKeys.includes(fieldKey) && fieldKeys.indexOf(fieldKey) === index,
    );
};
