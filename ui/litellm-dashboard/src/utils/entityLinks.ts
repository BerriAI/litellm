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

export function accessGroupHref(accessGroup: string): string {
  return `${uiHref("models-and-endpoints")}?access_group=${encodeURIComponent(accessGroup)}`;
}

export function modelOrAccessGroupHref(
  name: string,
  accessGroupNames: ReadonlySet<string> | undefined,
): string | undefined {
  if (accessGroupNames === undefined) return undefined;
  return accessGroupNames.has(name) ? accessGroupHref(name) : modelGroupHref(name);
}

export type ActivityScope = "key" | "team" | "user" | "model";

export interface ActivityLinkContext {
  from: string;
  to: string;
  alert?: string;
}

export function entityActivityHref(scope: ActivityScope, id: string, context: ActivityLinkContext): string | undefined {
  if (!id.trim()) return undefined;
  const unsafeKey = id.trim().startsWith("sk-") || id === "unattributed";
  if (scope === "key" && unsafeKey) return undefined;
  let href: string | undefined;
  switch (scope) {
    case "key":
      href = keyDetailHref(id);
      break;
    case "team":
      href = teamDetailHref(id);
      break;
    case "user":
      href = userDetailHref(id);
      break;
    case "model":
      href = `${uiHref("models-and-endpoints")}?activity_model=${encodeURIComponent(id)}`;
      break;
  }
  if (!href) return undefined;
  const params = new URLSearchParams({ detail_tab: "activity", activity_from: context.from, activity_to: context.to });
  if (context.alert) params.set("alert", context.alert);
  return `${href}&${params}`;
}
