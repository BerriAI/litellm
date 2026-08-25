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

export type Guardrail = components["schemas"]["GuardrailInfoResponse"];
export type GuardrailLitellmParams = components["schemas"]["LitellmParams"];
export type GuardrailMode = GuardrailLitellmParams["mode"];
export type GuardrailDefinitionLocation = components["schemas"]["GUARDRAIL_DEFINITION_LOCATION"];

export const GuardrailDefinitionLocation = {
  DB: "db",
  CONFIG: "config",
} as const satisfies Record<string, GuardrailDefinitionLocation>;
