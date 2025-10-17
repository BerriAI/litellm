import { Providers, provider_map, providerLogoMap } from "../provider_info_helpers";

export interface ProviderDisplayInfo {
  displayName: string;
  logo: string;
  enumKey: string | null;
}

/**
 * Convert backend provider value (e.g., "openai") to display info
 */
export const getProviderDisplayInfo = (providerValue: string): ProviderDisplayInfo => {
  const enumKey = Object.keys(provider_map).find(
    (key) => provider_map[key as keyof typeof provider_map] === providerValue
  );
  
  if (enumKey) {
    const displayName = Providers[enumKey as keyof typeof Providers];
    const logo = providerLogoMap[displayName];
    return { displayName, logo, enumKey };
  }
  
  return { displayName: providerValue, logo: "", enumKey: null };
};

/**
 * Convert provider enum key (e.g., "OpenAI") to backend value (e.g., "openai")
 */
export const getProviderBackendValue = (providerEnum: string): string | null => {
  return provider_map[providerEnum as keyof typeof provider_map] || null;
};

/**
 * Handle image error by replacing with fallback div
 */
export const handleImageError = (e: React.SyntheticEvent<HTMLImageElement>, fallbackText: string) => {
  const target = e.target as HTMLImageElement;
  const parent = target.parentElement;
  if (parent) {
    const fallbackDiv = document.createElement("div");
    fallbackDiv.className = "w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs";
    fallbackDiv.textContent = fallbackText.charAt(0);
    parent.replaceChild(fallbackDiv, target);
  }
};

