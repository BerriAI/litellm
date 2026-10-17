import type { ReactNode } from "react";
import { labelText, routeOf, sectionText, type MenuGroup } from "@/components/leftnav";

export type PaletteScope = "keys" | "global";

export interface PaletteNavItem {
  key: string;
  label: string;
  section: string;
  route: string;
  icon?: ReactNode;
}

export const paletteScopeForRoute = (routeSegment: string): PaletteScope =>
  routeSegment === "api-keys" ? "keys" : "global";

const matchRank = (item: PaletteNavItem, query: string): number => {
  const label = item.label.toLowerCase();
  if (label.startsWith(query)) return 0;
  if (label.includes(query)) return 1;
  if (item.section.toLowerCase().includes(query)) return 2;
  return -1;
};

export const flattenNavItems = (groups: readonly MenuGroup[]): PaletteNavItem[] =>
  groups.flatMap((group) => {
    const flattenItems = (items: MenuGroup["items"]): PaletteNavItem[] =>
      items.flatMap((item) => {
        if (item.external_url) return [];
        if (item.children) return flattenItems(item.children);
        return [
          {
            key: item.key,
            label: labelText(item),
            section: sectionText(group.groupLabel),
            route: routeOf(item),
            icon: item.icon,
          },
        ];
      });

    return flattenItems(group.items);
  });

export const matchNavItems = (items: readonly PaletteNavItem[], query: string): PaletteNavItem[] => {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) return [...items];

  return items
    .map((item, index) => {
      return { item, index, rank: matchRank(item, normalizedQuery) };
    })
    .filter(({ rank }) => rank >= 0)
    .sort((left, right) => left.rank - right.rank || left.index - right.index)
    .map(({ item }) => item);
};

export const shortcutLabel = (platform: string, userAgent: string): string =>
  [platform, userAgent].some((value) => value.toLowerCase().includes("mac")) ? "⌘K" : "Ctrl K";
