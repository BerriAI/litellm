import type { ParseKeys } from "i18next";
import { Providers } from "../provider_info_helpers";

export interface ProviderAuthType {
  readonly id: string;
  readonly labelKey: ParseKeys;
  readonly descriptionKey: ParseKeys;
  readonly fieldKeys: readonly string[];
  readonly requiredFieldKeys: readonly string[];
  readonly fixedValues?: Readonly<Record<string, string>>;
  readonly credentialOnly?: boolean;
}

const EMPTY_PROVIDER_AUTH_TYPES: readonly ProviderAuthType[] = [];

export const GITHUB_COPILOT_AUTH_TYPE_KEY = "github_copilot_auth_type";
export const GITHUB_COPILOT_HOST_KEY = "github_copilot_host";
export const GITHUB_COPILOT_PER_USER_AUTH_TYPE = "per_user_oauth";

export const PROVIDER_AUTH_TYPES: Partial<Record<keyof typeof Providers, readonly ProviderAuthType[]>> = {
  GITHUB_COPILOT: [
    {
      id: "shared_device_login",
      labelKey: "addModel.providerAuthTypes.sharedDeviceLogin.label",
      descriptionKey: "addModel.providerAuthTypes.sharedDeviceLogin.description",
      fieldKeys: ["api_base", "api_key"],
      requiredFieldKeys: [],
    },
    {
      id: GITHUB_COPILOT_PER_USER_AUTH_TYPE,
      labelKey: "addModel.providerAuthTypes.perUserOauth.label",
      credentialOnly: true,
      descriptionKey: "addModel.providerAuthTypes.perUserOauth.description",
      fieldKeys: [GITHUB_COPILOT_AUTH_TYPE_KEY, GITHUB_COPILOT_HOST_KEY],
      requiredFieldKeys: [GITHUB_COPILOT_AUTH_TYPE_KEY],
      fixedValues: { [GITHUB_COPILOT_AUTH_TYPE_KEY]: GITHUB_COPILOT_PER_USER_AUTH_TYPE },
    },
  ],
  MICROSOFT_365_COPILOT: [
    {
      id: "oauth_token_exchange",
      labelKey: "addModel.providerAuthTypes.oauthTokenExchange.label",
      credentialOnly: true,
      descriptionKey: "addModel.providerAuthTypes.oauthTokenExchange.description",
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
      labelKey: "addModel.providerAuthTypes.staticToken.label",
      descriptionKey: "addModel.providerAuthTypes.staticToken.description",
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
