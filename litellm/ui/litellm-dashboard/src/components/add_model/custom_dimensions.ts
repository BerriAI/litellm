import type { TFunction } from "i18next";
import { z } from "zod";
import i18n from "@/i18n/i18n";
import { DIMENSION_LABELS } from "./heuristic_scoring_knobs";

const customDimensionShape = {
  name: z.string(),
  weight: z.number(),
  keywords: z.array(z.string()).optional(),
  patterns: z.array(z.string()).optional(),
  scoring_mode: z.enum(["binary", "match_count"]).optional(),
};
const customDimensionSchema = z.object(customDimensionShape);

export type CustomDimension = z.infer<typeof customDimensionSchema>;
export type CustomDimensionRow = CustomDimension & { id: string };

export const hydrateCustomDimensions = (raw: unknown): CustomDimensionRow[] | undefined => {
  if (raw === undefined) return undefined;
  const parsed = z.array(customDimensionSchema).safeParse(raw);
  return parsed.success ? parsed.data.map((row, index) => ({ ...row, id: `stored-${index}` })) : undefined;
};

export const serializeCustomDimensions = (rows: CustomDimensionRow[]): CustomDimension[] =>
  rows.map(({ id: _id, ...dimension }) => dimension);

export const customDimensionsError = (
  rows: CustomDimensionRow[] | undefined,
  builtinNames: string[] = Object.keys(DIMENSION_LABELS),
  t: TFunction = i18n.t.bind(i18n),
): string | null => {
  if (!rows) return null;
  if (rows.length > 16) return t("addModel.customDimensions.tooMany");
  const names = rows.map((row) => row.name.toLowerCase());
  for (const [index, row] of rows.entries()) {
    const prefix = t("addModel.customDimensions.prefix", { number: index + 1 });
    if (!/^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(row.name)) return prefix + t("addModel.customDimensions.invalidName");
    if (builtinNames.some((name) => name.toLowerCase() === row.name.toLowerCase()))
      return prefix + t("addModel.customDimensions.builtInName");
    if (names.indexOf(row.name.toLowerCase()) !== index) return prefix + t("addModel.customDimensions.uniqueNames");
    if (!Number.isFinite(row.weight) || row.weight <= 0 || row.weight > 1)
      return prefix + t("addModel.customDimensions.weightRange");
    const matchers = [...(row.keywords ?? []), ...(row.patterns ?? [])];
    if (!matchers.length || matchers.some((matcher) => !matcher.trim()))
      return prefix + t("addModel.customDimensions.matcherRequired");
    if (
      matchers.length > 32 ||
      matchers.some((matcher) => [...matcher].length > 256) ||
      matchers.reduce((total, matcher) => total + [...matcher].length, 0) > 4096
    )
      return prefix + t("addModel.customDimensions.matcherLimits");
  }
  return null;
};
