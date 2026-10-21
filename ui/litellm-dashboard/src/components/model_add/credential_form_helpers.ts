import type { MountedFormValues } from "../common_components/MountedFormField";
import type { CredentialItem } from "../networking";

interface CredentialFormAdapter {
  getFieldValue: (field: string) => unknown;
  resetFields: () => void;
  setFieldValue: (field: string, value: unknown) => void;
}

/**
 * Reset the credential form when the user switches providers.
 *
 * Why: provider-specific fields (api_base, api_key, organization, ...)
 * share a single Antd Form state across providers. Without this reset,
 * the previous provider's values stick around — most visibly, OpenAI's
 * default `api_base` (https://api.openai.com/v1) carries over when the
 * user switches to Google AI Studio, overriding that provider's own
 * default_value.
 *
 * Strategy: blow away the whole form, then restore the provider-agnostic
 * fields (credential name + the new provider id) so the newly rendered
 * `ProviderSpecificFields` can apply its own defaults from a clean slate.
 *
 * The credential name is preserved because it's a user-supplied label
 * that shouldn't reset just because the admin re-selected a provider.
 */
const restrictedFields: readonly string[] = ["credential_name", "display_name", "custom_llm_provider"];

export type CredentialSubmission = {
  readonly credential_name: string;
  readonly display_name?: string | null;
  readonly custom_llm_provider: string;
  readonly credential_values: Record<string, unknown>;
};

export const buildCredential = (submission: CredentialSubmission, credentialValues: Record<string, unknown>) => ({
  credential_name: submission.credential_name,
  ...(submission.display_name !== undefined ? { display_name: submission.display_name } : {}),
  credential_values: credentialValues,
  credential_info: {
    custom_llm_provider: submission.custom_llm_provider,
  },
});

export const displayNameChange = (
  value: unknown,
  existingCredential: CredentialItem | null | undefined,
): { display_name?: string | null } => {
  const typed = typeof value === "string" ? value : "";
  if (!existingCredential) {
    return typed ? { display_name: typed } : {};
  }
  if (typed === (existingCredential.display_name ?? "")) {
    return {};
  }
  return { display_name: typed || null };
};

export const initialFormValues = (
  existingCredential: CredentialItem | null | undefined,
  initialProvider: string | null | undefined,
): MountedFormValues | undefined => {
  if (existingCredential) {
    return {
      ...Object.fromEntries(
        Object.entries(existingCredential.credential_values || {}).map(([key, value]) => [key, value ?? null]),
      ),
      credential_name: existingCredential.credential_name,
      display_name: existingCredential.display_name ?? "",
      custom_llm_provider: existingCredential.credential_info.custom_llm_provider,
    };
  }
  return initialProvider ? { custom_llm_provider: initialProvider } : undefined;
};

export const withoutRestrictedFields = (values: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(Object.entries(values).filter(([key]) => !restrictedFields.includes(key)));

export function resetCredentialFormOnProviderChange(
  form: CredentialFormAdapter,
  newProvider: string | null,
  setSelectedProvider: (p: string | null) => void,
): void {
  const preservedName = form.getFieldValue("credential_name");
  const preservedDisplayName = form.getFieldValue("display_name");
  form.resetFields();
  if (preservedName !== undefined) {
    form.setFieldValue("credential_name", preservedName);
  }
  if (preservedDisplayName !== undefined) {
    form.setFieldValue("display_name", preservedDisplayName);
  }
  setSelectedProvider(newProvider);
  form.setFieldValue("custom_llm_provider", newProvider);
}
