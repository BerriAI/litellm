import { createTabRoutes } from "@/utils/tabRoutes";

export const modelsRoutes = createTabRoutes("models-and-endpoints", [
  "add",
  "llm-credentials",
  "pass-through",
  "health",
  "retry-settings",
  "model-group-alias",
  "price-data",
] as const);

export type ModelTabSlug = (typeof modelsRoutes.slugs)[number];

export const MODELS_BASE_SEGMENT = modelsRoutes.baseSegment;
export const MODEL_TAB_SLUGS = modelsRoutes.slugs;
export const modelTabHref = modelsRoutes.tabHref;
export const slugFromPathname = modelsRoutes.slugFromPathname;
