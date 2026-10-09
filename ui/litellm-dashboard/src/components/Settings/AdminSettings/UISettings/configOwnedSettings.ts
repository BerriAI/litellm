import { z } from "zod";

const sourceByKeySchema = z.object({ source: z.record(z.string(), z.string()) });

export const parseConfigOwnedUISettings = (uiSettingsResponse: unknown): ReadonlySet<string> => {
  const parsed = sourceByKeySchema.safeParse(uiSettingsResponse);
  if (!parsed.success) return new Set();
  return new Set(
    Object.entries(parsed.data.source)
      .filter(([, source]) => source === "config")
      .map(([key]) => key),
  );
};
