import type { TFunction } from "i18next";
import { z } from "zod";

import { CACHE_FIELDS, type RedisType } from "./cacheSettingsFields";

const ADVANCED_SECTIONS = ["ssl", "cacheManagement", "gcp"] as const;
const cacheFormSchema = z.record(z.string(), z.union([z.string(), z.number(), z.boolean()]).nullish());

export type CacheFormValues = z.infer<typeof cacheFormSchema>;
export type CacheFormValue = CacheFormValues[string];

export const createCacheSettingsSchema = (redisType: RedisType, advancedOpen: boolean, t: TFunction) =>
  cacheFormSchema.superRefine((values, context) => {
    for (const field of CACHE_FIELDS) {
      if (field.redisType !== null && field.redisType !== redisType) continue;
      if (!advancedOpen && ADVANCED_SECTIONS.some((section) => section === field.section)) continue;

      const message = field.rules?.map((rule) => rule(values[field.name])).find((result) => result !== null);
      if (message) context.addIssue({ code: "custom", path: [field.name], message: t(message) });
    }
  });
