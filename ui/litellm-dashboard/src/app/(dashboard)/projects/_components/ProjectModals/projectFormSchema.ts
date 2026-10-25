import { z } from "zod";
import { englishTranslate, type Translate } from "@/i18n/language";

export const ALL_TEAM_MODELS = "all-team-models";

const repeatsEarlierValue = (values: readonly string[], index: number): boolean =>
  values[index] !== "" && values.indexOf(values[index]) !== index;

const modelLimitSchema = (t: Translate) =>
  z.object({
    model: z.string().min(1, t("Missing model")),
    tpm: z.number().optional(),
    rpm: z.number().optional(),
    itpm: z.number().optional(),
    otpm: z.number().optional(),
  });

export const buildProjectFormSchema = (t: Translate = englishTranslate) =>
  z
    .object({
      project_alias: z.string().min(1, t("Please enter a project name")),
      team_id: z
        .string()
        .nullable()
        .pipe(z.string({ error: t("Please select a team") }).min(1, t("Please select a team"))),
      description: z.string().optional(),
      models: z.array(z.string()),
      max_budget: z.number().nullish(),
      isBlocked: z.boolean(),
      guardrails: z.array(z.string()).optional(),
      modelLimits: z.array(modelLimitSchema(t)).optional(),
      metadata: z
        .array(
          z.object({
            key: z.string().min(1, t("Missing key")),
            value: z.string().min(1, t("Missing value")),
          }),
        )
        .optional(),
    })
    .superRefine((values, ctx) => {
      const models = (values.modelLimits ?? []).map((entry) => entry.model);
      models.forEach((_, index) => {
        if (repeatsEarlierValue(models, index)) {
          ctx.addIssue({ code: "custom", message: t("Duplicate model"), path: ["modelLimits", index, "model"] });
        }
      });

      const keys = (values.metadata ?? []).map((entry) => entry.key);
      keys.forEach((_, index) => {
        if (repeatsEarlierValue(keys, index)) {
          ctx.addIssue({ code: "custom", message: t("Duplicate key"), path: ["metadata", index, "key"] });
        }
      });
    });

export const projectFormSchema = buildProjectFormSchema();

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
