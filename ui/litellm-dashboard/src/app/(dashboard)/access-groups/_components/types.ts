export interface AccessGroup {
  id: string;
  name: string;
  description: string;
  modelIds: string[];
  mcpServerIds: string[];
  agentIds: string[];
  accessRoutes: string[];
  keyIds: string[];
  teamIds: string[];
  createdAt: string;
  createdBy: string;
  updatedAt: string;
  updatedBy: string;
}
