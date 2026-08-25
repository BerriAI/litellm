import { Providers } from "../provider_info_helpers";

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
export function resetCredentialFormOnProviderChange(
  form: CredentialFormAdapter,
  newProvider: Providers,
  setSelectedProvider: (p: Providers) => void,
): void {
  const preservedName = form.getFieldValue("credential_name");
  form.resetFields();
  if (preservedName !== undefined) {
    form.setFieldValue("credential_name", preservedName);
  }
  setSelectedProvider(newProvider);
  form.setFieldValue("custom_llm_provider", newProvider);
}

/**
 * Keys to drop from a saved credential's `credential_values` on update: whatever the form had
 * mounted before that it does not have mounted now. A field stops being mounted either because
 * the operator cleared it or because a credential_variants switch (e.g. api_key -> Keycloak)
 * unmounted it, and either way the backend must actually delete it rather than merge over it
 * -- a leftover field from a different auth variant fails validation on the next request
 * (wif.py rejects foreign-variant fields by presence).
 *
 * `mountedValues` must be the full projected form state (masked-but-untouched fields included),
 * not the caller's post-filter payload: a masked value that the operator never touched is still
 * mounted and must be preserved, not read as "absent, so delete it". A masked value is a
 * non-empty string, which is what separates it from a field the operator emptied.
 *
 * A cleared field stays mounted carrying an empty value, so emptiness counts as a deletion too.
 * Without that it is neither deleted here nor sent in the update (the caller drops empty values
 * from the payload), and the old value survives: clearing a destination would leave the previous
 * one still receiving requests that now carry a minted federation token.
 */
export function computeCredentialValuesToDelete(
  originalValues: Record<string, unknown>,
  mountedValues: Record<string, unknown>,
): string[] {
  return Object.keys(originalValues).filter((key) => {
    if (!(key in mountedValues)) return true;
    const value = mountedValues[key];
    return value === "" || value === null || value === undefined;
  });
}
