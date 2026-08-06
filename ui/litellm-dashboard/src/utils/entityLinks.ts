import { migratedHref } from "@/utils/migratedPages";

export function teamDetailHref(teamId: string): string {
  return `${migratedHref("teams")}?team=${encodeURIComponent(teamId)}`;
}

export function keyDetailHref(keyToken: string): string {
  return `${migratedHref("api-keys")}?key=${encodeURIComponent(keyToken)}`;
}
