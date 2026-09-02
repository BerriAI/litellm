export interface CapabilityCandidate {
  model: string;
  description: string;
}

export interface CapabilityRouterConfigValue {
  candidates: CapabilityCandidate[];
  classifier: { model: string; timeout_ms: number; max_output_tokens: number };
  probability_threshold: number;
  fallback_model: string;
  estimated_output_tokens: number;
  cache_ttl_seconds: number;
}

export const defaultCapabilityRouterConfig = (): CapabilityRouterConfigValue => ({
  candidates: [
    { model: "", description: "" },
    { model: "", description: "" },
  ],
  classifier: { model: "", timeout_ms: 3000, max_output_tokens: 1024 },
  probability_threshold: 0.7,
  fallback_model: "",
  estimated_output_tokens: 1000,
  cache_ttl_seconds: 3600,
});

export const capabilityRouterConfigError = (config: CapabilityRouterConfigValue): string | null => {
  if (config.candidates.length < 2) return "Add at least two candidate models";
  if (config.candidates.some((candidate) => !candidate.model.trim())) return "Select a model for every candidate";
  if (new Set(config.candidates.map((candidate) => candidate.model)).size !== config.candidates.length) {
    return "Candidate models must be unique";
  }
  if (config.candidates.some((candidate) => !candidate.description.trim())) {
    return "Describe what every candidate model is good at";
  }
  if (!config.classifier.model.trim()) return "Select a classifier model";
  if (!config.candidates.some((candidate) => candidate.model === config.fallback_model)) {
    return "Select one candidate as the fallback model";
  }
  if (
    !Number.isFinite(config.probability_threshold) ||
    config.probability_threshold < 0 ||
    config.probability_threshold > 1
  ) {
    return "Probability threshold must be between 0 and 1";
  }
  if (!Number.isInteger(config.cache_ttl_seconds) || config.cache_ttl_seconds < 1) {
    return "Cache TTL must be at least one second";
  }
  return null;
};

export const hydrateCapabilityRouterConfig = (stored: unknown): CapabilityRouterConfigValue => {
  const defaults = defaultCapabilityRouterConfig();
  if (typeof stored !== "object" || stored === null || Array.isArray(stored)) return defaults;
  const value = stored as Partial<CapabilityRouterConfigValue>;
  return {
    ...defaults,
    ...value,
    candidates: Array.isArray(value.candidates) ? value.candidates : defaults.candidates,
    classifier: { ...defaults.classifier, ...(value.classifier ?? {}) },
  };
};
