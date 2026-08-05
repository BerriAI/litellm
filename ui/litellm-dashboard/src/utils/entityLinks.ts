import { migratedHref } from "@/utils/migratedPages";

export function teamDetailHref(teamId: string): string {
  return `${migratedHref("teams")}?team=${encodeURIComponent(teamId)}`;
}

export function keyDetailHref(keyTokenHash: string): string {
  return `${migratedHref("api-keys")}?key=${encodeURIComponent(keyTokenHash)}`;
}

export function userDetailHref(userId: string): string {
  return `${migratedHref("users")}?user=${encodeURIComponent(userId)}`;
}

export function orgDetailHref(organizationId: string): string {
  return `${migratedHref("organizations")}?org=${encodeURIComponent(organizationId)}`;
}

export function modelDetailHref(modelId: string): string {
  return `${migratedHref("models-and-endpoints")}?model=${encodeURIComponent(modelId)}`;
}
