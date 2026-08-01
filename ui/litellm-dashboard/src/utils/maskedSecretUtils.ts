// The proxy redacts secrets in API responses by masking them (e.g. "sk-1****2345"),
// not by removing them. Edit forms must never echo a masked value back on save: the
// backend would encrypt the asterisks and overwrite the real secret. A run of 2+ mask
// chars only appears in masker output (real config -- incl. wildcard model names like
// "openai/*" -- carries at most a single "*"), so this reliably detects a redacted
// value without a provider-metadata lookup.
export const isMaskedSecret = (value: unknown): boolean => typeof value === "string" && /\*{2,}/.test(value);

export const stripMaskedSecrets = (params: Record<string, unknown>): Record<string, unknown> =>
  Object.fromEntries(Object.entries(params).filter(([, value]) => !isMaskedSecret(value)));
