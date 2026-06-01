export interface ModelNameValidationResult {
  isMisnamed: boolean;
  suggestions: string[];
}

const stripProviderPrefix = (value: string): string =>
  value.includes("/") ? value.split("/").slice(1).join("/") : value;

export const validateModelName = (
  modelName: string | undefined | null,
  providerModels: string[],
): ModelNameValidationResult => {
  const empty: ModelNameValidationResult = { isMisnamed: false, suggestions: [] };

  if (!modelName || providerModels.length === 0) {
    return empty;
  }

  const trimmed = modelName.trim();
  if (!trimmed) {
    return empty;
  }

  const lowerTrimmed = trimmed.toLowerCase();
  if (providerModels.some((known) => known.toLowerCase() === lowerTrimmed)) {
    return empty;
  }

  const needle = stripProviderPrefix(lowerTrimmed);
  if (!needle) {
    return empty;
  }

  const suggestions = providerModels.filter((known) => {
    const haystack = stripProviderPrefix(known.toLowerCase());
    return (
      haystack === needle ||
      haystack.startsWith(`${needle}-`) ||
      haystack.startsWith(`${needle}.`) ||
      haystack.startsWith(`${needle}_`) ||
      haystack.startsWith(`${needle}/`)
    );
  });

  return suggestions.length > 0 ? { isMisnamed: true, suggestions: suggestions.slice(0, 5) } : empty;
};
