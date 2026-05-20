// PROTOTYPE ONLY: header-variable persistence stored in localStorage.
// This is throwaway code used for a UI mockup and is replaced by real
// server-side storage in any production implementation.

export type VariableScope = "global" | "per_user";

export interface HeaderVariable {
  name: string;
  value: string;
  scope: VariableScope;
}

const VARIABLES_KEY = "litellm-mcp-proto-variables";
const USER_FIELDS_KEY = "litellm-mcp-proto-user-fields";

interface ServerLike {
  server_id?: string | null;
  alias?: string | null;
  server_name?: string | null;
}

export function serverKeyFor(server: ServerLike): string {
  return server.server_id || server.alias || server.server_name || "";
}

function readMap<T>(key: string): Record<string, T> {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as Record<string, T>) : {};
  } catch {
    return {};
  }
}

function writeMap<T>(key: string, value: Record<string, T>): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(key, JSON.stringify(value));
}

export function getServerVariables(serverKey: string): HeaderVariable[] {
  if (!serverKey) return [];
  const map = readMap<HeaderVariable[]>(VARIABLES_KEY);
  return map[serverKey] ?? [];
}

export function setServerVariables(serverKey: string, variables: HeaderVariable[]): void {
  if (!serverKey) return;
  const map = readMap<HeaderVariable[]>(VARIABLES_KEY);
  const cleaned = variables.filter((v) => v?.name?.trim());
  if (cleaned.length === 0) {
    delete map[serverKey];
  } else {
    map[serverKey] = cleaned;
  }
  writeMap(VARIABLES_KEY, map);
}

export function getUserFieldValues(serverKey: string): Record<string, string> {
  if (!serverKey) return {};
  const map = readMap<Record<string, string>>(USER_FIELDS_KEY);
  return map[serverKey] ?? {};
}

export function setUserFieldValues(serverKey: string, values: Record<string, string>): void {
  if (!serverKey) return;
  const map = readMap<Record<string, string>>(USER_FIELDS_KEY);
  map[serverKey] = values;
  writeMap(USER_FIELDS_KEY, map);
}

export function getMissingUserFields(server: ServerLike): string[] {
  const key = serverKeyFor(server);
  const altKey: string = server.alias && server.alias !== key ? server.alias : "";
  const primary = getServerVariables(key);
  const variables = primary.length > 0 || !altKey ? primary : getServerVariables(altKey);
  const userValues = { ...getUserFieldValues(altKey), ...getUserFieldValues(key) };
  return variables
    .filter((v) => v.scope === "per_user")
    .filter((v) => {
      const val = userValues[v.name];
      return !val || !val.trim();
    })
    .map((v) => v.name);
}

export function getAllVariablesFor(server: ServerLike): HeaderVariable[] {
  const key = serverKeyFor(server);
  const altKey: string = server.alias && server.alias !== key ? server.alias : "";
  const primary = getServerVariables(key);
  if (primary.length > 0) return primary;
  return altKey ? getServerVariables(altKey) : primary;
}
