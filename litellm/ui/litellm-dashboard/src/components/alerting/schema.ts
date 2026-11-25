import { z } from "zod";

export const alertingFormSchema = z.record(z.string(), z.union([z.string(), z.boolean()]));

export type AlertingFormValues = z.infer<typeof alertingFormSchema>;
