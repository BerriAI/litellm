import { createTabRoutes } from "@/utils/tabRoutes";

export const routerSettingsRoutes = createTabRoutes("router-settings", [
  "routing-groups",
  "fallbacks",
  "prompt-caching",
  "general",
] as const);

export type RouterSettingsTabSlug = (typeof routerSettingsRoutes.slugs)[number];
