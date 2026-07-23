import { migratedHref } from "@/utils/migratedPages";

export const MODELS_BASE_SEGMENT = "models-and-endpoints";

export const MODEL_TAB_SLUGS = [
  "add",
  "llm-credentials",
  "pass-through",
  "health",
  "retry-settings",
  "model-group-alias",
  "price-data",
] as const;

export type ModelTabSlug = (typeof MODEL_TAB_SLUGS)[number];

export function modelTabHref(slug: string): string {
  const base = migratedHref(MODELS_BASE_SEGMENT);
  return slug ? `${base}/${slug}/` : `${base}/`;
}

export function slugFromPathname(pathname: string): string {
  const parts = pathname.split("/").filter(Boolean);
  const idx = parts.indexOf(MODELS_BASE_SEGMENT);
  if (idx === -1) {
    return "";
  }
  return parts[idx + 1] ?? "";
}
