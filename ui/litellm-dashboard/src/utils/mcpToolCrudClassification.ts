export type CrudOp = "read" | "create" | "update" | "delete" | "unknown";

const DELETE_RE = /\b(delete|remove|destroy|purge|drop|clear|erase|unlink|disconnect)\b/i;
const CREATE_RE = /\b(create|add|insert|new|post|submit|register|make|generate|write|upload)\b/i;
const UPDATE_RE = /\b(update|edit|modify|change|patch|put|set|rename|move|transform)\b/i;
const READ_RE = /\b(get|read|list|fetch|search|find|query|retrieve|show|view|check|describe|info)\b/i;

export interface MCPToolEntry {
  name: string;
  description?: string;
}

export function classifyToolOp(name: string, description = ""): CrudOp {
  const text = (name + " " + description).toLowerCase();
  if (DELETE_RE.test(text)) return "delete";
  if (UPDATE_RE.test(text)) return "update";
  if (CREATE_RE.test(text)) return "create";
  if (READ_RE.test(text)) return "read";
  return "unknown";
}

export function groupToolsByCrud(tools: MCPToolEntry[]): Record<CrudOp, MCPToolEntry[]> {
  const groups: Record<CrudOp, MCPToolEntry[]> = {
    read: [],
    create: [],
    update: [],
    delete: [],
    unknown: [],
  };
  for (const tool of tools) {
    const op = classifyToolOp(tool.name, tool.description);
    groups[op].push(tool);
  }
  return groups;
}

export const CRUD_GROUP_META: Record<
  CrudOp,
  { label: string; description: string; risk: "low" | "medium" | "high" | "unknown"; color: string }
> = {
  read: {
    label: "Read",
    description: "Safe operations — fetch, list, search. No side effects.",
    risk: "low",
    color: "green",
  },
  create: {
    label: "Create",
    description: "Add new resources — insert, upload, register.",
    risk: "medium",
    color: "blue",
  },
  update: {
    label: "Update",
    description: "Modify existing resources — edit, patch, rename.",
    risk: "medium",
    color: "yellow",
  },
  delete: {
    label: "Delete",
    description: "Destructive operations — remove, purge, destroy.",
    risk: "high",
    color: "red",
  },
  unknown: {
    label: "Other",
    description: "Operations that could not be automatically classified.",
    risk: "unknown",
    color: "gray",
  },
};
