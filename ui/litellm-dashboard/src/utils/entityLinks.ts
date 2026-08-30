import { migratedHref } from "@/utils/migratedPages";

const MODEL_GRANT_SENTINELS: ReadonlySet<string> = new Set([
  "all-proxy-models",
  "all-team-models",
  "no-default-models",
]);

export function teamDetailHref(teamId: string): string {
  return `${migratedHref("teams")}?team=${encodeURIComponent(teamId)}`;
}

export function keyDetailHref(keyToken: string): string {
  return `${migratedHref("api-keys")}?key=${encodeURIComponent(keyToken)}`;
}

export function userDetailHref(userId: string): string {
  return `${migratedHref("users")}?user=${encodeURIComponent(userId)}`;
}

export function orgDetailHref(orgId: string): string {
  return `${migratedHref("organizations")}?org=${encodeURIComponent(orgId)}`;
}

export function modelGroupHref(modelGroup: string): string | undefined {
  if (MODEL_GRANT_SENTINELS.has(modelGroup)) return undefined;
  return `${migratedHref("models-and-endpoints")}?model_group=${encodeURIComponent(modelGroup)}`;
}
