export interface Project {
  id: string;
  name: string;
  description: string;
  teamId: string;
  teamAlias: string;
  models: string[];
  status: "active" | "blocked";
  spend: number;
  createdAt: string;
  createdBy: string;
  updatedAt: string;
  updatedBy: string;
}
