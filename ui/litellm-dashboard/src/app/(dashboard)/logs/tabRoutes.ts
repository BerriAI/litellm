import { createTabRoutes } from "@/utils/tabRoutes";

export const logsRoutes = createTabRoutes("logs", ["audit", "deleted-keys", "deleted-teams"] as const);

export type LogsTabSlug = (typeof logsRoutes.slugs)[number];
