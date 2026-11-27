import type { TFunction } from "i18next";
import { z } from "zod";
import {
  GUARDRAIL_STREAM_SCOPES,
  type GuardrailStreamScope,
  type LoggingOnlyScopeChoice,
} from "../guardrail_info_helpers";

const loggingOnlyScopeChoiceSchema = z.enum(["default", "input", "output"]) satisfies z.ZodType<LoggingOnlyScopeChoice>;

const streamScopeSchema = z.enum(GUARDRAIL_STREAM_SCOPES) satisfies z.ZodType<GuardrailStreamScope>;

export const createCustomCodeSchema = (t: TFunction) => {
  const shape = {
    guardrail_name: z.string().refine((v) => v.trim().length > 0, {
      message: t("guardrails.customCode.toast.nameRequired"),
    }),
    mode: z.array(z.string()),
    stream_scope_by_mode: z.record(z.string(), streamScopeSchema),
    logging_only_scope_choice: loggingOnlyScopeChoiceSchema,
    logging_only_continue: z.boolean(),
    default_on: z.boolean(),
    code: z.string().refine((v) => v.trim().length > 0, {
      message: t("guardrails.customCode.toast.codeRequired"),
    }),
  };
  return z.object(shape);
};

export type CustomCodeFormValues = z.infer<ReturnType<typeof createCustomCodeSchema>>;
