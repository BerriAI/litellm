export enum GuardrailProviders {
    PresidioPII = "Presidio PII",
    Bedrock = "Bedrock Guardrail",
    Lakera = "Lakera",
    AzureContentSafetyPromptShield = "Azure Content Safety Prompt Shield",
    AzureContentSafetyTextModeration = "Azure Content Safety Text Moderation"
}

export const guardrail_provider_map: Record<string, string> = {
    PresidioPII: "presidio",
    Bedrock: "bedrock",
    Lakera: "lakera_v2",
    AzureContentSafetyPromptShield: "azure/prompt_shield",
    AzureContentSafetyTextModeration: "azure/text_moderations"
};


// Decides if we should render the PII config settings for a given provider
// For now we only support PII config settings for Presidio PII
export const shouldRenderPIIConfigSettings = (provider: string | null) => {
    if (!provider) {
        return false;
    }
    // cast provider to GuardrailProviders enum
    const providerEnum = GuardrailProviders[provider as keyof typeof GuardrailProviders];
    return providerEnum === GuardrailProviders.PresidioPII;
};

// Decides if we should render the Azure Text Moderation config settings for a given provider
export const shouldRenderAzureTextModerationConfigSettings = (provider: string | null) => {
    if (!provider) {
        return false;
    }
    // cast provider to GuardrailProviders enum
    const providerEnum = GuardrailProviders[provider as keyof typeof GuardrailProviders];
    return providerEnum === GuardrailProviders.AzureContentSafetyTextModeration;
};

const asset_logos_folder = '../ui/assets/logos/';

export const guardrailLogoMap: Record<string, string> = {
    [GuardrailProviders.PresidioPII]: `${asset_logos_folder}presidio.png`,
    [GuardrailProviders.Bedrock]: `${asset_logos_folder}bedrock.svg`,
    [GuardrailProviders.Lakera]: `${asset_logos_folder}lakeraai.jpeg`,
    [GuardrailProviders.AzureContentSafetyPromptShield]: `${asset_logos_folder}presidio.png`,
    [GuardrailProviders.AzureContentSafetyTextModeration]: `${asset_logos_folder}presidio.png`,
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

    // Get the display name from GuardrailProviders enum and logo from map
    const displayName = GuardrailProviders[enumKey as keyof typeof GuardrailProviders];
    const logo = guardrailLogoMap[displayName as keyof typeof guardrailLogoMap];

    return { logo, displayName };
};
