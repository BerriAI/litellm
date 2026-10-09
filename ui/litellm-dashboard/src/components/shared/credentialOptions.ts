import type { SearchSelectOption } from "@/components/shared/SearchSelect";
import type { CredentialItem } from "@/components/networking";

export const NO_CREDENTIAL_OPTION: SearchSelectOption = { label: "None", value: "" };

export const credentialLabel = (credential: CredentialItem): string =>
  credential.display_name || credential.credential_name;

export const credentialLabelsByName = (credentials: CredentialItem[]): ReadonlyMap<string, string> =>
  new Map(credentials.map((credential) => [credential.credential_name, credentialLabel(credential)]));

export const toCredentialOption = (credential: CredentialItem): SearchSelectOption => ({
  label: credentialLabel(credential),
  value: credential.credential_name,
  sublabel: credential.display_name ? credential.credential_name : undefined,
});

export const credentialOptions = (credentials: CredentialItem[]): SearchSelectOption[] => [
  NO_CREDENTIAL_OPTION,
  ...credentials.map(toCredentialOption),
];
