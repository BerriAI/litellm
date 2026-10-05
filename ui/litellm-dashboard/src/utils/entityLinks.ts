import { DEFAULT_PROXY_ADMIN_USER_ID, UI_TEAM_ID } from "@/utils/sentinels";
import { uiHref } from "@/utils/uiHref";

const MODEL_GRANT_SENTINELS: ReadonlySet<string> = new Set([
  "all-proxy-models",
  "all-team-models",
  "no-default-models",
]);

export function teamDetailHref(teamId: string): string | undefined {
  if (teamId === UI_TEAM_ID) return undefined;
  return `${uiHref("teams")}?team=${encodeURIComponent(teamId)}`;
}

export function keyDetailHref(keyToken: string): string {
  return `${uiHref("api-keys")}?key=${encodeURIComponent(keyToken)}`;
}

export function userDetailHref(userId: string): string | undefined {
  if (userId === DEFAULT_PROXY_ADMIN_USER_ID) return undefined;
  return `${uiHref("users")}?user=${encodeURIComponent(userId)}`;
}

export function orgDetailHref(orgId: string): string {
  return `${uiHref("organizations")}?org=${encodeURIComponent(orgId)}`;
}

export function modelGroupHref(modelGroup: string): string | undefined {
  if (MODEL_GRANT_SENTINELS.has(modelGroup)) return undefined;
  return `${uiHref("models-and-endpoints")}?model_group=${encodeURIComponent(modelGroup)}`;
}
