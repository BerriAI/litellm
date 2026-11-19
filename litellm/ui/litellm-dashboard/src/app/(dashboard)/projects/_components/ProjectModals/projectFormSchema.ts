import { z } from "zod";

import i18n from "@/i18n/i18n";

export const ALL_TEAM_MODELS = "all-team-models";

const repeatsEarlierValue = (values: readonly string[], index: number): boolean =>
  values[index] !== "" && values.indexOf(values[index]) !== index;

const modelLimitSchema = z.object({
  model: z.string().min(1, { error: () => i18n.t("projects.form.errors.missingModel") }),
  tpm: z.number().optional(),
  rpm: z.number().optional(),
  itpm: z.number().optional(),
  otpm: z.number().optional(),
  batchEnqueuedTokens: z
    .number()
    .int({ error: () => i18n.t("projects.form.errors.batchEnqueuedTokens") })
    .positive({ error: () => i18n.t("projects.form.errors.batchEnqueuedTokens") })
    .optional(),
});

export const projectFormSchema = z
  .object({
    project_alias: z.string().min(1, { error: () => i18n.t("projects.form.errors.projectNameRequired") }),
    team_id: z
      .string()
      .nullable()
      .pipe(
        z
          .string({ error: () => i18n.t("projects.form.errors.teamRequired") })
          .min(1, { error: () => i18n.t("projects.form.errors.teamRequired") }),
      ),
    description: z.string().optional(),
    models: z.array(z.string()),
    max_budget: z.number().nullish(),
    isBlocked: z.boolean(),
    guardrails: z.array(z.string()).optional(),
    modelLimits: z.array(modelLimitSchema).optional(),
    metadata: z
      .array(
        z.object({
          key: z.string().min(1, { error: () => i18n.t("projects.form.errors.missingKey") }),
          value: z.string().min(1, { error: () => i18n.t("projects.form.errors.missingValue") }),
        }),
      )
      .optional(),
  })
  .superRefine((values, ctx) => {
    const models = (values.modelLimits ?? []).map((entry) => entry.model);
    models.forEach((_, index) => {
      if (repeatsEarlierValue(models, index)) {
        ctx.addIssue({
          code: "custom",
          message: i18n.t("projects.form.errors.duplicateModel"),
          path: ["modelLimits", index, "model"],
        });
      }
    });

    const keys = (values.metadata ?? []).map((entry) => entry.key);
    keys.forEach((_, index) => {
      if (repeatsEarlierValue(keys, index)) {
        ctx.addIssue({
          code: "custom",
          message: i18n.t("projects.form.errors.duplicateKey"),
          path: ["metadata", index, "key"],
        });
      }
    });
  });

export type ProjectFormValues = z.input<typeof projectFormSchema>;
export type ProjectSubmitValues = z.output<typeof projectFormSchema>;

export const emptyProjectFormValues: ProjectFormValues = {
  project_alias: "",
  team_id: null,
  description: undefined,
  models: [],
  max_budget: undefined,
  isBlocked: false,
  guardrails: undefined,
  modelLimits: undefined,
  metadata: undefined,
};
