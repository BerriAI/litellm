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

export interface Guardrail {
  guardrail_id: string;
  guardrail_name: string | null;
  litellm_params: {
    guardrail: string;
    mode: string;
    default_on: boolean;
    pii_entities_config?: { [key: string]: string };
    [key: string]: any;
  };
  guardrail_info: Record<string, any> | null;
  created_at?: string;
  updated_at?: string;
}
