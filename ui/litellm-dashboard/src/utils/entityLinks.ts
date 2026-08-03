import { migratedHref } from "@/utils/migratedPages";

export function teamDetailHref(teamId: string): string {
  return `${migratedHref("teams")}?team=${encodeURIComponent(teamId)}`;
}
