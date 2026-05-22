/**
 * PROTOTYPE-ONLY mock storage for the MCP "user fields" feature.
 *
 * Real implementation will store defs server-side (per MCP server) and
 * encrypted per-user values in the DB / vault. This file fakes defs in
 * localStorage so admins can keep editing them across sessions, and keeps
 * per-user values (which may contain secrets like bearer tokens) in
 * sessionStorage so they do not survive browser close.
 *
 * Do NOT model real auth/credential storage after this file.
 */

export interface UserField {
  name: string;
  label: string;
  description?: string;
  secret?: boolean;
}

const DEFS_KEY = (serverId: string) => `mcp_user_fields_defs_${serverId}`;
const VALUES_KEY = (serverId: string, userId: string) =>
  `mcp_user_fields_values_${serverId}_${userId}`;

function safeParse<T>(raw: string | null, fallback: T): T {
  if (!raw) return fallback;
  try {
    return JSON.parse(raw) as T;
  } catch {
    return fallback;
  }
}

export function getUserFieldDefs(serverId: string): UserField[] {
  if (typeof window === "undefined") return [];
  return safeParse<UserField[]>(window.localStorage.getItem(DEFS_KEY(serverId)), []);
}

export function setUserFieldDefs(serverId: string, fields: UserField[]): void {
  if (typeof window === "undefined") return;
  if (!fields || fields.length === 0) {
    window.localStorage.removeItem(DEFS_KEY(serverId));
    return;
  }
  window.localStorage.setItem(DEFS_KEY(serverId), JSON.stringify(fields));
}

export function getUserFieldValues(
  serverId: string,
  userId: string,
): Record<string, string> {
  if (typeof window === "undefined") return {};
  return safeParse<Record<string, string>>(
    window.sessionStorage.getItem(VALUES_KEY(serverId, userId)),
    {},
  );
}

export function setUserFieldValues(
  serverId: string,
  userId: string,
  values: Record<string, string>,
): void {
  if (typeof window === "undefined") return;
  window.sessionStorage.setItem(
    VALUES_KEY(serverId, userId),
    JSON.stringify(values),
  );
}

export function removeUserFieldDefs(serverId: string): void {
  if (typeof window === "undefined") return;
  window.localStorage.removeItem(DEFS_KEY(serverId));
}

export function getMissingUserFields(
  serverId: string,
  userId: string,
): UserField[] {
  const defs = getUserFieldDefs(serverId);
  if (defs.length === 0) return [];
  const values = getUserFieldValues(serverId, userId);
  return defs.filter(
    (f) =>
      !values[f.name] ||
      typeof values[f.name] !== "string" ||
      values[f.name].trim() === "",
  );
}
