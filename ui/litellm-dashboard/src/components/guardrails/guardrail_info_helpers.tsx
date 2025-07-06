// Legacy enum - keeping for backward compatibility
export enum GuardrailProviders {
    PresidioPII = "Presidio PII",
    Bedrock = "Bedrock Guardrail",
    Lakera = "Lakera",
}

// Dynamic guardrail providers object - populated from API response
export let DynamicGuardrailProviders: Record<string, string> = {};

// Function to populate dynamic providers from API response
export const populateGuardrailProviders = (providerParamsResponse: Record<string, any>) => {
    const providers: Record<string, string> = {};
    
    // Legacy hardcoded providers for backward compatibility
    providers.PresidioPII = "Presidio PII";
    providers.Bedrock = "Bedrock Guardrail";
    providers.Lakera = "Lakera";
    
    // Add dynamic providers from API response
    Object.entries(providerParamsResponse).forEach(([key, value]) => {
        if (value && typeof value === 'object' && 'ui_friendly_name' in value) {
            // Create a key from the provider name (camelCase)
            const providerKey = key.split('_').map((word, index) => 
                index === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : 
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join('');
            
            providers[providerKey] = value.ui_friendly_name;
        }
    });
    
    DynamicGuardrailProviders = providers;
    return providers;
};

// Function to get current guardrail providers (dynamic or fallback to legacy)
export const getGuardrailProviders = () => {
    return Object.keys(DynamicGuardrailProviders).length > 0 ? DynamicGuardrailProviders : GuardrailProviders;
};

export const guardrail_provider_map: Record<string, string> = {
    PresidioPII: "presidio",
    Bedrock: "bedrock",
    Lakera: "lakera_v2",
};

// Function to populate provider map from API response - updates the original map
export const populateGuardrailProviderMap = (providerParamsResponse: Record<string, any>) => {
    // Add dynamic providers from API response directly to the main map
    Object.entries(providerParamsResponse).forEach(([key, value]) => {
        if (value && typeof value === 'object' && 'ui_friendly_name' in value) {
            // Create a key from the provider name (camelCase)
            const providerKey = key.split('_').map((word, index) => 
                index === 0 ? word.charAt(0).toUpperCase() + word.slice(1) : 
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join('');
            
            guardrail_provider_map[providerKey] = key; // Add directly to the main map
        }
    });
};

// Decides if we should render the PII config settings for a given provider
// For now we only support PII config settings for Presidio PII
export const shouldRenderPIIConfigSettings = (provider: string | null) => {
    if (!provider) {
        return false;
    }
    // Check both dynamic and legacy providers
    const currentProviders = getGuardrailProviders();
    const providerEnum = currentProviders[provider as keyof typeof currentProviders];
    return providerEnum === "Presidio PII";
};

// Decides if we should render the Azure Text Moderation config settings for a given provider
export const shouldRenderAzureTextModerationConfigSettings = (provider: string | null) => {
    if (!provider) {
        return false;
    }
    // Check both dynamic and legacy providers
    const currentProviders = getGuardrailProviders();
    const providerEnum = currentProviders[provider as keyof typeof currentProviders];
    return providerEnum === "Azure Content Safety Text Moderation";
};

const asset_logos_folder = '../ui/assets/logos/';

export const guardrailLogoMap: Record<string, string> = {
    "Presidio PII": `${asset_logos_folder}presidio.png`,
    "Bedrock Guardrail": `${asset_logos_folder}bedrock.svg`,
    "Lakera": `${asset_logos_folder}lakeraai.jpeg`,
    "Azure Content Safety Prompt Shield": `${asset_logos_folder}presidio.png`,
    "Azure Content Safety Text Moderation": `${asset_logos_folder}presidio.png`,
};

export const getGuardrailLogoAndName = (guardrailValue: string): { logo: string, displayName: string } => {
    if (!guardrailValue) {
        return { logo: "", displayName: "-" };
    }

    // Find the enum key by matching guardrail_provider_map values
    const enumKey = Object.keys(guardrail_provider_map).find(
        key => guardrail_provider_map[key].toLowerCase() === guardrailValue.toLowerCase()
    );

    if (!enumKey) {
        return { logo: "", displayName: guardrailValue };
    }

    // Get the display name from current GuardrailProviders and logo from map
    const currentProviders = getGuardrailProviders();
    const displayName = currentProviders[enumKey as keyof typeof currentProviders];
    const logo = guardrailLogoMap[displayName as keyof typeof guardrailLogoMap];

    return { logo: logo || "", displayName: displayName || guardrailValue };
};
