import { migratedHref } from "@/utils/migratedPages";

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
