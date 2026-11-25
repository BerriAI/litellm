import { z } from "zod";

export const loggingSettingsSchema = z.object({
  store_prompts_in_spend_logs: z.boolean(),
  maximum_spend_logs_retention_period: z.string(),
  maximum_spend_logs_cleanup_batch_size: z.string(),
  maximum_spend_logs_cleanup_max_batches: z.string(),
  maximum_spend_logs_cleanup_run_budget: z.string(),
  maximum_spend_logs_cleanup_batch_timeout: z.string(),
});

export type LoggingSettingsFormValues = z.infer<typeof loggingSettingsSchema>;
