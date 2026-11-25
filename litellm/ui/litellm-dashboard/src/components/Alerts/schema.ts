import { z } from "zod";

const thresholdFieldsSchema = z.object({
  multiplier: z.string(),
  minimum: z.string(),
});

export const alertThresholdsSchema = z.object({
  spend: thresholdFieldsSchema,
  traffic: thresholdFieldsSchema,
  latency: thresholdFieldsSchema,
  reliability: thresholdFieldsSchema,
});

export type AlertThresholdsFields = z.infer<typeof alertThresholdsSchema>;
