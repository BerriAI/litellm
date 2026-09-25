export type CrudOp = "read" | "create" | "update" | "delete" | "unknown";

const READ_TOKENS = new Set([
  "get",
  "read",
  "list",
  "fetch",
  "search",
  "find",
  "query",
  "retrieve",
  "show",
  "view",
  "check",
  "describe",
  "info",
  "lookup",
  "count",
  "export",
  "download",
]);
const DELETE_TOKENS = new Set([
  "delete",
  "remove",
  "destroy",
  "purge",
  "drop",
  "erase",
  "unlink",
  "wipe",
  "clear",
  "revoke",
  "uninstall",
  "trash",
  "truncate",
  "rm",
  "del",
]);
const UPDATE_TOKENS = new Set([
  "update",
  "edit",
  "modify",
  "change",
  "patch",
  "put",
  "set",
  "rename",
  "move",
  "transform",
  "toggle",
  "enable",
  "disable",
  "archive",
  "restore",
]);
const CREATE_TOKENS = new Set([
  "create",
  "add",
  "insert",
  "new",
  "post",
  "submit",
  "register",
  "make",
  "generate",
  "write",
  "upload",
  "send",
  "publish",
]);

const SPLIT_RE = /[_\-./\s]+/;
const CAMEL_BOUNDARY_RE = /(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])/;
const NON_WORD_RE = /[^\w]+/;

export interface MCPToolEntry {
  name: string;
  description?: string;
}

const nameTokens = (name: string): string[] =>
  name
    .split(SPLIT_RE)
    .flatMap((chunk) => chunk.split(CAMEL_BOUNDARY_RE))
    .filter((token) => token.length > 0)
    .map((token) => token.toLowerCase());

const descriptionTokens = (description: string): string[] =>
  description
    .split(NON_WORD_RE)
    .filter((token) => token.length > 0)
    .map((token) => token.toLowerCase());

const stripSuffix = (token: string, suffix: string): string => (token.endsWith(suffix) ? token.slice(0, -suffix.length) : token);

const tokenVariants = (token: string): string[] =>
  [token, stripSuffix(token, "es"), stripSuffix(token, "s"), stripSuffix(token, "ed"), stripSuffix(token, "ing")].filter(
    (variant) => variant.length > 0,
  );

// Matches litellm/proxy/_experimental/mcp_server/tool_classification.py, fixture-pinned.
const classifyTokens = (tokens: string[]): CrudOp => {
  const variants = new Set(tokens.flatMap((token) => tokenVariants(token)));
  if ([...variants].some((variant) => READ_TOKENS.has(variant))) return "read";
  if ([...variants].some((variant) => DELETE_TOKENS.has(variant))) return "delete";
  if ([...variants].some((variant) => UPDATE_TOKENS.has(variant))) return "update";
  if ([...variants].some((variant) => CREATE_TOKENS.has(variant))) return "create";
  return "unknown";
};

export function classifyToolOp(name: string, description = ""): CrudOp {
  const byName = classifyTokens(nameTokens(name));
  if (byName !== "unknown") return byName;
  if (description) return classifyTokens(descriptionTokens(description));
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
  { label: string; description: string; risk: "low" | "medium" | "high" | "unknown" }
> = {
  read: {
    label: "Read",
    description: "Safe operations — fetch, list, search. No side effects.",
    risk: "low",
  },
  create: {
    label: "Create",
    description: "Add new resources — insert, upload, register.",
    risk: "medium",
  },
  update: {
    label: "Update",
    description: "Modify existing resources — edit, patch, rename.",
    risk: "medium",
  },
  delete: {
    label: "Delete",
    description: "Destructive operations — remove, purge, destroy.",
    risk: "high",
  },
  unknown: {
    label: "Other",
    description: "Operations that could not be automatically classified.",
    risk: "unknown",
  },
};
