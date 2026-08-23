import type {
  ProviderCredentialFieldMetadata,
  ProviderCredentialVariant,
  ProviderCredentialVariants,
} from "../networking";

export const getVariant = (
  variants: ProviderCredentialVariants,
  variantId: string,
): ProviderCredentialVariant | undefined => variants.variants.find((variant) => variant.id === variantId);

export const resolveVariantFieldDefs = (
  variants: ProviderCredentialVariants,
  variantId: string,
): ProviderCredentialFieldMetadata[] => {
  const variant = getVariant(variants, variantId);
  if (!variant) {
    return [];
  }
  const byKey = new Map(variants.field_definitions.map((field) => [field.key, field]));
  return variant.field_keys
    .map((key) => byKey.get(key))
    .filter((field): field is ProviderCredentialFieldMetadata => field !== undefined);
};

const hasValue = (value: unknown): boolean => (typeof value === "string" ? value.trim() !== "" : value != null);

const isFullySatisfied = (
  variant: ProviderCredentialVariant,
  fieldsByKey: Map<string, ProviderCredentialFieldMetadata>,
  values: Record<string, unknown>,
): boolean => {
  const fixedValuesMatch = Object.entries(variant.fixed_values).every(([key, value]) => values[key] === value);
  if (!fixedValuesMatch) {
    return false;
  }
  return variant.field_keys.every((key) => !fieldsByKey.get(key)?.required || hasValue(values[key]));
};

/**
 * Infers which variant a set of existing field values belongs to, for pre-selecting the
 * selector when editing a saved credential. Checks variants in declaration order and keeps
 * the last one whose fixed_values match and whose required fields are all present, so a more
 * specific variant (one with a matching discriminator, or more required fields set) wins over
 * a broader one earlier in the list (e.g. a bare api_key variant with no required fields,
 * which always matches trivially). Falls back to default_variant when nothing matches, which
 * is also what a blank/new form resolves to.
 */
export const inferActiveVariant = (variants: ProviderCredentialVariants, values: Record<string, unknown>): string => {
  const fieldsByKey = new Map(variants.field_definitions.map((field) => [field.key, field]));
  return variants.variants.reduce(
    (best, variant) => (isFullySatisfied(variant, fieldsByKey, values) ? variant.id : best),
    variants.default_variant,
  );
};
