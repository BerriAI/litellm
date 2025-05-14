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
  selectedActions: {[key: string]: string};
  onEntitySelect: (entity: string) => void;
  onActionSelect: (entity: string, action: string) => void;
  entityCategories?: PiiEntityCategory[];
} 