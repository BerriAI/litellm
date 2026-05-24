// PROTOTYPE: mocked env-var storage for the "MCP per-user fields" demo.
// All state lives in localStorage; no backend wiring. Throwaway code — once
// the customer agrees on the flow, this gets rebuilt against the real DB.

export type EnvVarScope = "instance" | "per_user";

export interface EnvVarDefinition {
  name: string;
  value: string;
  scope: EnvVarScope;
}

const defsKey = (alias: string) => `mock-mcp-env-defs::${alias}`;
const userKey = (alias: string, userId: string) =>
  `mock-mcp-env-user::${alias}::${userId}`;

export function getEnvVarDefinitions(serverAlias: string): EnvVarDefinition[] {
  if (!serverAlias || typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(defsKey(serverAlias));
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    // Migrate legacy "global" scope to "instance" so the dropdown stays in sync
    // with renamed labels without forcing a re-save.
    return parsed.map((d: EnvVarDefinition) =>
      d && (d.scope as unknown) === "global"
        ? { ...d, scope: "instance" as const }
        : d,
    );
  } catch {
    return [];
  }
}

export function setEnvVarDefinitions(
  serverAlias: string,
  defs: EnvVarDefinition[],
): void {
  if (!serverAlias || typeof window === "undefined") return;
  try {
    window.localStorage.setItem(defsKey(serverAlias), JSON.stringify(defs));
  } catch (err) {
    console.warn("[mock-mcp-env-vars] failed to save defs", err);
  }
}

export function getPerUserValues(
  serverAlias: string,
  userId: string,
): Record<string, string> {
  if (!serverAlias || !userId || typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(userKey(serverAlias, userId));
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

export function setPerUserValues(
  serverAlias: string,
  userId: string,
  values: Record<string, string>,
): void {
  if (!serverAlias || !userId || typeof window === "undefined") return;
  try {
    window.localStorage.setItem(
      userKey(serverAlias, userId),
      JSON.stringify(values),
    );
  } catch (err) {
    console.warn("[mock-mcp-env-vars] failed to save user values", err);
  }
}

export function getMissingUserFields(
  serverAlias: string,
  userId: string,
): string[] {
  const defs = getEnvVarDefinitions(serverAlias);
  const values = getPerUserValues(serverAlias, userId);
  return defs
    .filter((d) => d.scope === "per_user")
    .map((d) => d.name)
    .filter((name) => !values[name] || values[name].trim() === "");
}

export function getPerUserFieldNames(serverAlias: string): string[] {
  return getEnvVarDefinitions(serverAlias)
    .filter((d) => d.scope === "per_user")
    .map((d) => d.name);
}

// Bump on each save so list views can re-read localStorage without polling.
const CHANGE_EVENT = "mock-mcp-env-vars-changed";

export function notifyEnvVarsChanged(): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new CustomEvent(CHANGE_EVENT));
}

export function subscribeEnvVarsChanged(handler: () => void): () => void {
  if (typeof window === "undefined") return () => undefined;
  window.addEventListener(CHANGE_EVENT, handler);
  // also react to storage events from other tabs
  const storageHandler = (e: StorageEvent) => {
    if (e.key && e.key.startsWith("mock-mcp-env-")) handler();
  };
  window.addEventListener("storage", storageHandler);
  return () => {
    window.removeEventListener(CHANGE_EVENT, handler);
    window.removeEventListener("storage", storageHandler);
  };
}
