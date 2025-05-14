export enum GuardrailProviders {
    PresidioPII = "Presidio PII",
    Aporia = "Aporia",
    AimSecurity = "Aim Security",
    Bedrock = "Amazon Bedrock",
    GuardrailsAI = "Guardrails.ai",
    LakeraAI = "Lakera AI",
    Custom = "Custom Guardrail",
    PromptInjection = "Prompt Injection Detection",
}

export const guardrail_provider_map: Record<string, string> = {
    Aporia: "aporia",
    AimSecurity: "aim",
    Bedrock: "bedrock",
    GuardrailsAI: "guardrails_ai",
    LakeraAI: "lakera",
    PromptInjection: "detect_prompt_injection",
    PresidioPII: "presidio",
};

const asset_logos_folder = '../ui/assets/logos/';

export const guardrailLogoMap: Record<string, string> = {
    [GuardrailProviders.Aporia]: `${asset_logos_folder}aporia.svg`,
    [GuardrailProviders.AimSecurity]: `${asset_logos_folder}aim.svg`,
    [GuardrailProviders.Bedrock]: `${asset_logos_folder}bedrock.svg`,
    [GuardrailProviders.GuardrailsAI]: `${asset_logos_folder}guardrails_ai.svg`,
    [GuardrailProviders.LakeraAI]: `${asset_logos_folder}lakera.svg`,
    [GuardrailProviders.PromptInjection]: `${asset_logos_folder}prompt_injection.svg`,
    [GuardrailProviders.PresidioPII]: `${asset_logos_folder}presidio.svg`
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
