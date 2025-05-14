export enum GuardrailProviders {
    PresidioPII = "Presidio PII",
}

export const guardrail_provider_map: Record<string, string> = {
    PresidioPII: "presidio",
};

// Define which providers need specific fields
export const provider_specific_fields: Record<string, string[]> = {
    PresidioPII: ["pii_entities", "pii_actions"]
};

const asset_logos_folder = '../ui/assets/logos/';

export const guardrailLogoMap: Record<string, string> = {
    [GuardrailProviders.PresidioPII]: `${asset_logos_folder}presidio.png`
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
