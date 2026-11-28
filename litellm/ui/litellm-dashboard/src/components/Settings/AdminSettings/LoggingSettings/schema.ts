import { z } from "zod";

const loggingSettingsFields = {
  store_prompts_in_spend_logs: z.unknown(),
  maximum_spend_logs_retention_period: z.string(),
  maximum_spend_logs_cleanup_batch_size: z.string(),
  maximum_spend_logs_cleanup_max_batches: z.string(),
  maximum_spend_logs_cleanup_run_budget: z.string(),
  maximum_spend_logs_cleanup_batch_timeout: z.string(),
  guardrail_log_skip_passed: z.boolean(),
};

export const loggingSettingsSchema = z.object(loggingSettingsFields);

export type LoggingSettingsFormValues = z.infer<typeof loggingSettingsSchema>;
