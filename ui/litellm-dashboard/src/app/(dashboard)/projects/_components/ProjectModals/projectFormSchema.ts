import { z } from "zod/v4";

export const ALL_TEAM_MODELS = "all-team-models";

const repeatsEarlierValue = (values: readonly string[], index: number): boolean =>
  values[index] !== "" && values.indexOf(values[index]) !== index;

const modelLimitSchema = z.object({
  model: z.string().min(1, "Missing model"),
  tpm: z.number().optional(),
  rpm: z.number().optional(),
  itpm: z.number().optional(),
  otpm: z.number().optional(),
});

export const projectFormSchema = z
  .object({
    project_alias: z.string().min(1, "Please enter a project name"),
    team_id: z.string().min(1, "Please select a team"),
    description: z.string().optional(),
    models: z.array(z.string()),
    max_budget: z.number().optional(),
    isBlocked: z.boolean(),
    guardrails: z.array(z.string()).optional(),
    modelLimits: z.array(modelLimitSchema).optional(),
    metadata: z
      .array(
        z.object({
          key: z.string().min(1, "Missing key"),
          value: z.string().min(1, "Missing value"),
        }),
      )
      .optional(),
  })
  .superRefine((values, ctx) => {
    const models = (values.modelLimits ?? []).map((entry) => entry.model);
    models.forEach((_, index) => {
      if (repeatsEarlierValue(models, index)) {
        ctx.addIssue({ code: "custom", message: "Duplicate model", path: ["modelLimits", index, "model"] });
      }
    });

    const keys = (values.metadata ?? []).map((entry) => entry.key);
    keys.forEach((_, index) => {
      if (repeatsEarlierValue(keys, index)) {
        ctx.addIssue({ code: "custom", message: "Duplicate key", path: ["metadata", index, "key"] });
      }
    });
  });

export type ProjectFormValues = z.output<typeof projectFormSchema>;

export const emptyProjectFormValues: ProjectFormValues = {
  project_alias: "",
  team_id: "",
  description: undefined,
  models: [],
  max_budget: undefined,
  isBlocked: false,
  guardrails: undefined,
  modelLimits: undefined,
  metadata: undefined,
};
