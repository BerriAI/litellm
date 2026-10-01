import type { SearchSelectOption } from "@/components/shared/SearchSelect";
import type { CredentialItem } from "@/components/networking";

export const NO_CREDENTIAL_OPTION: SearchSelectOption = { label: "None", value: "" };

export const toCredentialOption = (credential: CredentialItem): SearchSelectOption => ({
  label: credential.credential_name,
  value: credential.credential_name,
  sublabel: credential.credential_alias ?? undefined,
});

export const credentialOptions = (credentials: CredentialItem[]): SearchSelectOption[] => [
  NO_CREDENTIAL_OPTION,
  ...credentials.map(toCredentialOption),
];
