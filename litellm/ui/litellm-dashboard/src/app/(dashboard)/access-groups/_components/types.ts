export interface AccessGroup {
  id: string;
  name: string;
  description: string;
  modelIds: string[];
  mcpServerIds: string[];
  agentIds: string[];
  skillNames: string[];
  vectorStoreIds: string[];
  keyIds: string[];
  teamIds: string[];
  createdAt: string;
  createdBy: string;
  updatedAt: string;
  updatedBy: string;
}
