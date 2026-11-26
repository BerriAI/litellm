import { z } from "zod";

export const submissionRulesSchema = z.object({
  requiredFields: z.custom<string[]>(Array.isArray),
});

export type SubmissionRulesValues = z.infer<typeof submissionRulesSchema>;
