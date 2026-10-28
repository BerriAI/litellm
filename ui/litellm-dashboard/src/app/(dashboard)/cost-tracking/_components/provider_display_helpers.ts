import { provider_map } from "@/components/provider_info_helpers";

/**
 * Convert provider enum key (e.g., "OpenAI") to backend value (e.g., "openai")
 */
export const getProviderBackendValue = (providerEnum: string): string | null => {
  return provider_map[providerEnum as keyof typeof provider_map] || null;
};
