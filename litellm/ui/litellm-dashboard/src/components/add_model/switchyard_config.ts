import { z } from "zod";

const threshold = z.number().min(0).max(1);
const positiveInteger = z.number().int().positive();
const modelName = z
  .string()
  .trim()
  .min(1, "Select a model")
  .refine((name) => !name.startsWith("auto_router/"), "Select a concrete model");
const optionsFields = {
  picker: z.enum(["efficient_first", "capable_first"]).default("efficient_first"),
  confidence_threshold: threshold.nullish(),
  recent_window: positiveInteger.nullish(),
  tool_semantics: z.partialRecord(z.enum(["observe", "mutate", "plan", "new"]), z.array(z.string())).nullish(),
  only_on_wrong_signal_escalation: z.boolean().optional(),
};
const llmClassifierFields = {
  kind: z.literal("llm").default("llm"),
  base_threshold: threshold.default(0.5),
  threshold_step: threshold.default(0),
  max_output_tokens: positiveInteger.default(4096),
  recent_turn_window: positiveInteger.nullish(),
  response_format_type: z.enum(["json_schema", "json_object"]).default("json_schema"),
  prompt: z.string().nullish(),
};
const decisionClassifierFields = {
  kind: z.literal("decision"),
  cutoff: threshold.default(0.6),
  instructions: z.union([z.string(), z.record(z.string(), z.unknown())]).nullish(),
  evidence: z.record(z.string(), z.unknown()).nullish(),
};
const configFields = {
  algorithm: z.literal("stage_router").default("stage_router"),
  models: z.object({ capable: modelName, efficient: modelName, judge: modelName.nullish() }).strict(),
  options: z.object(optionsFields).strict().default({ picker: "efficient_first" }),
  classifier: z.union([z.object(decisionClassifierFields).strict(), z.object(llmClassifierFields).strict()]).nullish(),
};
export const switchyardConfigSchema = z
  .object(configFields)
  .strict()
  .superRefine((value, ctx) => {
    if (Boolean(value.classifier) !== Boolean(value.models.judge)) {
      ctx.addIssue({
        code: "custom",
        message: "Select a judge when the classifier is enabled",
        path: ["models", "judge"],
      });
    }
  });

export type SwitchyardConfig = z.infer<typeof switchyardConfigSchema>;
export type ClassifierKind = NonNullable<SwitchyardConfig["classifier"]>["kind"];
export const isDecisionModelMode = (mode: string | undefined): boolean => mode === "decisions" || mode === "evaluation";
export const defaultClassifier = (kind: ClassifierKind = "llm"): NonNullable<SwitchyardConfig["classifier"]> =>
  kind === "decision"
    ? { kind: "decision", cutoff: 0.6 }
    : {
        kind: "llm",
        base_threshold: 0.5,
        threshold_step: 0,
        max_output_tokens: 4096,
        response_format_type: "json_schema",
      };
export const emptySwitchyardConfig = (): SwitchyardConfig => ({
  algorithm: "stage_router",
  models: { capable: "", efficient: "" },
  options: { picker: "efficient_first" },
});
export const stageConfidence = (config: SwitchyardConfig): number =>
  config.options.confidence_threshold ?? (config.classifier ? 0.7 : 0.5);
export const toggleClassifier = (config: SwitchyardConfig, enabled: boolean): SwitchyardConfig => ({
  ...config,
  models: {
    capable: config.models.capable,
    efficient: config.models.efficient,
    ...(enabled ? { judge: config.models.judge ?? "" } : {}),
  },
  classifier: enabled ? config.classifier ?? defaultClassifier() : undefined,
});

export const setClassifierKind = (config: SwitchyardConfig, kind: ClassifierKind): SwitchyardConfig =>
  config.classifier?.kind === kind
    ? config
    : { ...config, models: { ...config.models, judge: "" }, classifier: defaultClassifier(kind) };

export function parseToolSemantics(text: string): SwitchyardConfig["options"]["tool_semantics"] {
  if (!text.trim()) return undefined;
  return optionsFields.tool_semantics.parse(JSON.parse(text));
}

export function switchyardSaveError(name: string, config: SwitchyardConfig): string | null {
  if (!name.trim()) return "Enter a router name";
  const validated = switchyardConfigSchema.safeParse(config);
  if (!validated.success) return validated.error.issues[0].message;
  if (Object.values(validated.data.models).includes(name.trim())) return "The router cannot select itself";
  return null;
}

export const switchyardRouterPayload = (name: string, config: SwitchyardConfig) => ({
  model_name: name.trim(),
  litellm_params: { model: "auto_router/switchyard", switchyard_router_config: switchyardConfigSchema.parse(config) },
});
