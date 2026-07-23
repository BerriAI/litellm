import { createTabRoutes } from "@/utils/tabRoutes";

export const cachingRoutes = createTabRoutes("caching", ["health", "settings", "coordination-redis"] as const);

export type CacheTabSlug = (typeof cachingRoutes.slugs)[number];
