export enum VectorStoreProviders {
    Bedrock = "Amazon Bedrock",
    PgVector = "PostgreSQL with pgvector"
}

export const vectorStoreProviderMap: Record<string, string> = {
    Bedrock: "bedrock",
    PgVector: "pg_vector"
};

const asset_logos_folder = '/assets/logos/';

export const vectorStoreProviderLogoMap: Record<string, string> = {
    [VectorStoreProviders.Bedrock]: `${asset_logos_folder}bedrock.svg`,
    [VectorStoreProviders.PgVector]: `${asset_logos_folder}postgresql.svg`, // Fallback to a generic database icon if needed
};

// Define field types for provider-specific configurations
export interface VectorStoreFieldConfig {
    name: string;
    label: string;
    tooltip: string;
    placeholder?: string;
    required: boolean;
    type?: 'text' | 'password';
}

// Provider-specific field configurations
export const vectorStoreProviderFields: Record<string, VectorStoreFieldConfig[]> = {
    bedrock: [],
    pg_vector: [
        {
            name: "api_base",
            label: "API Base",
            tooltip: "The base URL for your PostgreSQL database (e.g., http://localhost:5432)",
            placeholder: "http://localhost:5432",
            required: true,
            type: "text"
        },
        {
            name: "api_key", 
            label: "API Key",
            tooltip: "Database password or connection string for authentication",
            placeholder: "Database password or connection string",
            required: true,
            type: "password"
        }
    ]
};

export const getVectorStoreProviderLogoAndName = (providerValue: string): { logo: string, displayName: string } => {
    if (!providerValue) {
        return { logo: "", displayName: "-" };
    }

    // Find the enum key by matching vectorStoreProviderMap values
    const enumKey = Object.keys(vectorStoreProviderMap).find(
        key => vectorStoreProviderMap[key].toLowerCase() === providerValue.toLowerCase()
    );

    if (!enumKey) {
        return { logo: "", displayName: providerValue };
    }

    // Get the display name from VectorStoreProviders enum and logo from map
    const displayName = VectorStoreProviders[enumKey as keyof typeof VectorStoreProviders];
    const logo = vectorStoreProviderLogoMap[displayName as keyof typeof vectorStoreProviderLogoMap];

    return { logo, displayName };
};

export const getProviderSpecificFields = (providerValue: string): VectorStoreFieldConfig[] => {
    return vectorStoreProviderFields[providerValue] || [];
}; 