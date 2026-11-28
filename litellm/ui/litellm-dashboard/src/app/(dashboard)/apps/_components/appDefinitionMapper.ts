import type { AppResourceType } from "./appForms";

interface StoredDefinition {
  resource_type?: AppResourceType | null;
  auto_create_customers_without_app?: boolean;
}

export const initialAppDefinitionValues = { draft: null };

export const appDefinitionTypeBody = (resourceType: AppResourceType, definition: StoredDefinition | undefined) => ({
  resource_type: resourceType,
  auto_create_customers_without_app: definition?.auto_create_customers_without_app ?? true,
});

export const appDefinitionWithoutAppBody = (allowed: boolean, definition: StoredDefinition | undefined) => ({
  resource_type: definition?.resource_type ?? null,
  auto_create_customers_without_app: allowed,
});
