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
const restrictedFields: readonly string[] = ["credential_name", "custom_llm_provider"];

export const buildCredential = (values: Record<string, unknown>, credentialValues: Record<string, unknown>) => ({
  credential_name: values.credential_name as string,
  credential_values: credentialValues,
  credential_info: {
    custom_llm_provider: values.custom_llm_provider as string,
  },
});

export const withoutRestrictedFields = (values: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(Object.entries(values).filter(([key]) => !restrictedFields.includes(key)));

export function resetCredentialFormOnProviderChange(
  form: CredentialFormAdapter,
  newProvider: string | null,
  setSelectedProvider: (p: string | null) => void,
): void {
  const preservedName = form.getFieldValue("credential_name");
  form.resetFields();
  if (preservedName !== undefined) {
    form.setFieldValue("credential_name", preservedName);
  }
  setSelectedProvider(newProvider);
  form.setFieldValue("custom_llm_provider", newProvider);
}
