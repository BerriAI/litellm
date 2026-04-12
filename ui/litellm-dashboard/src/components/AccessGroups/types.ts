export interface AccessGroup {
  id: string
  name: string
  description: string
  modelIds: string[]
  mcpServerIds: string[]
  agentIds: string[]
  keyIds: string[]
  teamIds: string[]
  createdAt: string
  createdBy: string
  updatedAt: string
  updatedBy: string
}

export interface Model {
  id: string
  name: string
  provider: string
}

export interface McpServer {
  id: string
  name: string
  endpoint: string
}

export interface Agent {
  id: string
  name: string
  type: string
}

export interface AccessGroupKey {
  id: string
  alias: string
  status: string
  createdAt: string
}

export interface AccessGroupTeam {
  id: string
  name: string
  members: number
  role: string
}
