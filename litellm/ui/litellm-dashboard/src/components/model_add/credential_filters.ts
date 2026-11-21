import type { CredentialItem } from "@/components/networking";

export type ProviderLabel = (provider: string) => string;

export interface CredentialFilters {
  readonly query: string;
  readonly providers: readonly string[];
}

export interface CredentialProviderOption {
  readonly label: string;
  readonly value: string;
}

const providerOf = (credential: CredentialItem): string | undefined => credential.credential_info?.custom_llm_provider;

const haystack = (credential: CredentialItem, providerLabel: ProviderLabel): readonly string[] => {
  const provider = providerOf(credential);
  return [
    credential.credential_name,
    credential.display_name ?? "",
    provider ?? "",
    provider === undefined ? "" : providerLabel(provider),
  ];
};

const matchesCredentialQuery = (credential: CredentialItem, query: string, providerLabel: ProviderLabel): boolean => {
  const needle = query.trim().toLowerCase();
  if (needle === "") {
    return true;
  }
  return haystack(credential, providerLabel).some((value) => value.toLowerCase().includes(needle));
};

export const filterCredentials = (
  credentials: readonly CredentialItem[],
  { query, providers }: CredentialFilters,
  providerLabel: ProviderLabel,
): CredentialItem[] =>
  credentials.filter((credential) => {
    const provider = providerOf(credential);
    const providerMatches = providers.length === 0 || (provider !== undefined && providers.includes(provider));
    return providerMatches && matchesCredentialQuery(credential, query, providerLabel);
  });

export const credentialProviderOptions = (
  credentials: readonly CredentialItem[],
  providerLabel: ProviderLabel,
): CredentialProviderOption[] =>
  Array.from(new Set(credentials.map(providerOf)))
    .filter((provider): provider is string => provider !== undefined && provider !== "")
    .map((provider) => ({ value: provider, label: providerLabel(provider) }))
    .sort((a, b) => a.label.localeCompare(b.label) || a.value.localeCompare(b.value));
