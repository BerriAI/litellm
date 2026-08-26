import type { components } from "@/lib/http/schema";

export interface PiiEntity {
  name: string;
  category: string;
}

export interface PiiEntityCategory {
  category: string;
  entities: string[];
}

export interface PiiConfigurationProps {
  entities: string[];
  actions: string[];
  selectedEntities: string[];
  selectedActions: { [key: string]: string };
  onEntitySelect: (entity: string) => void;
  onActionSelect: (entity: string, action: string) => void;
  entityCategories?: PiiEntityCategory[];
}

// Partial because the read endpoints serialize with response_model_exclude_unset: only stored keys reach the wire
export type GuardrailLitellmParams = Partial<components["schemas"]["LitellmParams"]>;
export type Guardrail = Omit<components["schemas"]["GuardrailInfoResponse"], "litellm_params"> & {
  litellm_params?: GuardrailLitellmParams | null;
};
export type GuardrailMode = components["schemas"]["LitellmParams"]["mode"];
export type GuardrailDefinitionLocation = components["schemas"]["GUARDRAIL_DEFINITION_LOCATION"];

export const GuardrailDefinitionLocation = {
  DB: "db",
  CONFIG: "config",
} as const satisfies Record<string, GuardrailDefinitionLocation>;
