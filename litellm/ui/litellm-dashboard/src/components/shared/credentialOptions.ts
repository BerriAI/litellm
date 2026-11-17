import type { ParseKeys } from "i18next";

import type { SearchSelectOption } from "@/components/shared/SearchSelect";
import type { CredentialItem } from "@/components/networking";
import i18n from "@/i18n/i18n";

const NO_CREDENTIAL_LABEL: ParseKeys = "shared.credentialOptions.none";

export const NO_CREDENTIAL_OPTION: SearchSelectOption = { label: NO_CREDENTIAL_LABEL, value: "" };

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
  { ...NO_CREDENTIAL_OPTION, label: i18n.t(NO_CREDENTIAL_LABEL) },
  ...credentials.map(toCredentialOption),
];
