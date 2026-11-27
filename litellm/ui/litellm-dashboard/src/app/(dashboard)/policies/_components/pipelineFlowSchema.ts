import type { TFunction } from "i18next";
import { z } from "zod";
import type { GuardrailPipeline } from "@/components/policies/types";

export const createPipelineFlowSchema = (t: TFunction) =>
  z.object({
    policy_name: z.string().refine((v) => v.trim().length > 0, {
      message: t("policies.flowBuilder.page.toasts.nameRequired"),
    }),
    description: z.string(),
    pipeline: z
      .custom<GuardrailPipeline>()
      .refine((p) => Array.isArray(p?.steps) && p.steps.every((s) => Boolean(s.guardrail)), {
        message: t("policies.flowBuilder.page.toasts.guardrailRequired"),
      }),
  });

export type PipelineFlowFormValues = z.infer<ReturnType<typeof createPipelineFlowSchema>>;
