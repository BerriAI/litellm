import type { TFunction } from "i18next";
import { z } from "zod";

import { COORDINATION_FIELDS, type CoordinationRedisType } from "./coordinationRedisFields";

const coordinationFormSchema = z.record(z.string(), z.union([z.string(), z.number(), z.boolean()]).optional());

export type CoordinationFormValues = z.infer<typeof coordinationFormSchema>;
export type CoordinationFormValue = CoordinationFormValues[string];

export const createCoordinationRedisSchema = (redisType: CoordinationRedisType, t: TFunction) =>
  coordinationFormSchema.superRefine((values, context) => {
    for (const field of COORDINATION_FIELDS) {
      if (field.redisType !== null && field.redisType !== redisType) continue;

      const message = field.rules?.map((rule) => rule(values[field.name])).find((result) => result !== null);
      if (message) context.addIssue({ code: "custom", path: [field.name], message: t(message) });
    }
  });
