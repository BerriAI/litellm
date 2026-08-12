// The proxy redacts secrets in API responses by masking them (e.g. "sk-1****2345"),
// not by removing them. Edit forms must never echo a masked value back on save: the
// backend would encrypt the asterisks and overwrite the real secret. A run of 2+ mask
// chars only appears in masker output (real config -- incl. wildcard model names like
// "openai/*" -- carries at most a single "*"), so this reliably detects a redacted
// value without a provider-metadata lookup.
export const isMaskedSecret = (value: unknown): boolean => typeof value === "string" && /\*{2,}/.test(value);

export const stripMaskedSecrets = (params: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(Object.entries(params).filter(([, value]) => !isMaskedSecret(value)));

// The model edit form seeds its LiteLLM Params editor from the stored params minus the
// credential name and any masked secret (both preserved through other paths). A key that
// was in that seed but is absent from the params being saved was deleted by the user, so
// it must be sent as an explicit null to clear it — the backend PATCH merge is additive,
// so an omitted key keeps its stored value. Excluding masked secrets here is what stops a
// redacted-and-stripped credential from being mistaken for a deletion and nulled out.
export const computeRemovedLitellmParamKeys = (
  storedParams: Record<string, unknown>,
  savedParams: Record<string, unknown>,
): string[] =>
  Object.entries(storedParams)
    .filter(([key, value]) => key !== "litellm_credential_name" && !isMaskedSecret(value) && !(key in savedParams))
    .map(([key]) => key);
