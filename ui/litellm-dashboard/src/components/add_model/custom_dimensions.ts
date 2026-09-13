import { z } from "zod";
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
): string | null => {
  if (!rows) return null;
  if (rows.length > 16) return "A router can have at most 16 custom dimensions";
  const names = rows.map((row) => row.name.toLowerCase());
  for (const [index, row] of rows.entries()) {
    const prefix = `Custom dimension ${index + 1}: `;
    if (!/^[A-Za-z][A-Za-z0-9_]{0,63}$/.test(row.name))
      return (
        prefix + "use a name starting with a letter, followed by letters, numbers or underscores (64 characters max)"
      );
    if (builtinNames.some((name) => name.toLowerCase() === row.name.toLowerCase()))
      return prefix + "choose a name that is not already a built-in weight";
    if (names.indexOf(row.name.toLowerCase()) !== index) return prefix + "names must be unique";
    if (!Number.isFinite(row.weight) || row.weight <= 0 || row.weight > 1)
      return prefix + "weight must be greater than 0 and at most 1";
    const matchers = [...(row.keywords ?? []), ...(row.patterns ?? [])];
    if (!matchers.length || matchers.some((matcher) => !matcher.trim()))
      return prefix + "add at least one nonblank keyword or pattern";
    if (
      matchers.length > 32 ||
      matchers.some((matcher) => [...matcher].length > 256) ||
      matchers.reduce((total, matcher) => total + [...matcher].length, 0) > 4096
    )
      return prefix + "use at most 32 matchers, 256 characters each and 4096 characters combined";
  }
  return null;
};
