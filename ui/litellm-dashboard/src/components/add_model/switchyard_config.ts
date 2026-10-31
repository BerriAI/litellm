import { z } from "zod";

export const SWITCHYARD_ALGORITHMS = ["random", "capability", "custom", "escalation", "stage"] as const;
export type SwitchyardAlgorithm = (typeof SWITCHYARD_ALGORITHMS)[number];
export type SwitchyardDraft = Record<string, unknown>;

export const switchyardRecord = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

const jsonValue = (value: unknown): unknown => {
  if (typeof value !== "string") return value;
  try {
    return JSON.parse(value);
  } catch {
    return value;
  }
};
const positiveInteger = z.number().int().positive();
const optionalWindow = positiveInteger.nullable().optional();
const responseFormat = z.enum(["json_schema", "json_object"]).optional();
const optionalPrompt = z.string().nullable().optional();
const classifierFields = {
  session_affinity: z.boolean().optional(),
  message_hash_fallback: z.boolean().optional(),
  recent_turn_window: optionalWindow,
  max_output_tokens: positiveInteger.optional(),
};
const capabilityFields = {
  ...classifierFields,
  base_threshold: z.number().min(0).max(1).optional(),
  threshold_step: z.number().nonnegative().optional(),
  prompt: optionalPrompt,
  response_format_type: responseFormat,
};
const affinityError = (
  value: { message_hash_fallback?: boolean; session_affinity?: boolean },
  ctx: z.RefinementCtx,
) => {
  if (value.message_hash_fallback && !value.session_affinity)
    ctx.addIssue({
      code: "custom",
      path: ["message_hash_fallback"],
      message: "Message hash fallback requires session affinity",
    });
};
const capabilitySchema = z.object(capabilityFields).passthrough().superRefine(affinityError);
const deescalationFields = {
  strong_min_calls: positiveInteger,
  confirmations: positiveInteger,
  strong_max_calls: positiveInteger.nullable().optional(),
  weak_cooldown_calls: z.number().int().nonnegative().optional(),
};
const deescalationSchema = z
  .object(deescalationFields)
  .passthrough()
  .superRefine((value, ctx) => {
    if (value.strong_max_calls != null && value.strong_max_calls < value.strong_min_calls)
      ctx.addIssue({ code: "custom", message: "Maximum calls must be at least minimum calls" });
  });
const modelsSchema = z.record(z.string().min(1), z.array(z.string().trim().min(1)).min(1));
const escalationFields = {
  confirmations: positiveInteger.optional(),
  recent_turn_window: positiveInteger.optional(),
  window_message_chars: z.number().int().min(50).optional(),
  max_output_tokens: positiveInteger.optional(),
  prompt: optionalPrompt,
  response_format_type: responseFormat,
  deescalation: z.preprocess(jsonValue, deescalationSchema.nullable().optional()),
};
const customFields = {
  ...classifierFields,
  prompt: z.string().min(1).optional(),
  response_schema: z.preprocess(jsonValue, z.record(z.string(), z.unknown()).optional()),
  selector: z.string().min(1).optional(),
  default_target: z.string().min(1).optional(),
};
const stageFields = {
  picker: z.enum(["efficient_first", "capable_first"]).optional(),
  confidence_threshold: z.number().min(0).max(1).optional(),
  recent_window: optionalWindow,
  escalation_note: optionalPrompt,
  deescalation_note: optionalPrompt,
  only_on_wrong_signal_escalation: z.boolean().optional(),
  capable_system_prompt: optionalPrompt,
  efficient_system_prompt: optionalPrompt,
  tool_semantics: z.preprocess(jsonValue, z.record(z.string(), z.array(z.string())).nullable().optional()),
  classifier: z.preprocess(jsonValue, capabilitySchema.nullable().optional()),
};
const configFields = {
  algorithm: z.enum(SWITCHYARD_ALGORITHMS),
  models: z.preprocess(jsonValue, modelsSchema),
  weights: z.preprocess(jsonValue, z.array(z.number().nonnegative()).min(1).nullable().optional()),
  seed: z.number().int().nonnegative().nullable().optional(),
  fallback_enabled: z.boolean().optional(),
  capability: capabilitySchema.optional(),
  escalation: z.object(escalationFields).passthrough().optional(),
  custom: z.object(customFields).passthrough().superRefine(affinityError).nullable().optional(),
  stage: z.object(stageFields).passthrough().optional(),
};
export function switchyardRequiredGroups(algorithm: string, hasStageClassifier: boolean): string[] {
  if (algorithm === "random") return ["any"];
  if (algorithm === "custom") return ["judge"];
  if (algorithm === "stage" && !hasStageClassifier) return ["efficient", "capable"];
  return ["efficient", "capable", "judge"];
}
export const switchyardConfigSchema = z
  .object(configFields)
  .passthrough()
  .superRefine((config, ctx) => {
    for (const group of switchyardRequiredGroups(config.algorithm, Boolean(config.stage?.classifier))) {
      if (!config.models[group]?.length)
        ctx.addIssue({ code: "custom", path: ["models", group], message: `Select a model for ${group}` });
    }
    if (!Object.keys(config.models).length)
      ctx.addIssue({ code: "custom", path: ["models"], message: "Add at least one target group" });
    if (config.algorithm !== "random" || !config.weights) return;
    const validWeights =
      config.weights.length === config.models.any?.length && config.weights.some((weight) => weight > 0);
    if (!validWeights)
      ctx.addIssue({
        code: "custom",
        path: ["weights"],
        message: "Provide one weight per model in the any group, with at least one positive weight",
      });
  })
  .superRefine((config, ctx) => {
    if (config.algorithm !== "custom") return;
    for (const field of ["prompt", "response_schema", "selector", "default_target"] as const) {
      if (!config.custom?.[field])
        ctx.addIssue({ code: "custom", path: ["custom", field], message: `${field} is required for custom routing` });
    }
    if (config.custom?.default_target && !config.models[config.custom.default_target])
      ctx.addIssue({
        code: "custom",
        path: ["custom", "default_target"],
        message: "Default target must name a model group",
      });
  })
  .transform((config) =>
    config.algorithm === "random"
      ? config
      : Object.fromEntries(Object.entries(config).filter(([key]) => key !== "weights")),
  );

export const SWITCHYARD_DEFAULTS: Record<SwitchyardAlgorithm, Record<string, unknown>> = {
  random: {},
  capability: {
    base_threshold: 0.5,
    threshold_step: 0,
    session_affinity: false,
    message_hash_fallback: false,
    recent_turn_window: null,
    max_output_tokens: 4096,
    prompt: null,
    response_format_type: "json_schema",
  },
  escalation: {
    confirmations: 2,
    recent_turn_window: 28,
    window_message_chars: 500,
    max_output_tokens: 4096,
    prompt: null,
    response_format_type: "json_schema",
    deescalation: null,
  },
  custom: { session_affinity: false, message_hash_fallback: false, recent_turn_window: null, max_output_tokens: 4096 },
  stage: {
    picker: "efficient_first",
    confidence_threshold: 0.5,
    recent_window: null,
    escalation_note: null,
    deescalation_note: null,
    only_on_wrong_signal_escalation: true,
    capable_system_prompt: null,
    efficient_system_prompt: null,
    tool_semantics: null,
    classifier: null,
  },
};

export const newSwitchyardConfig = (): SwitchyardDraft => ({
  algorithm: "capability",
  models: {},
  fallback_enabled: true,
});
export const restoreSwitchyardDefaults = (config: SwitchyardDraft): SwitchyardDraft => {
  const algorithm = config.algorithm as SwitchyardAlgorithm;
  if (algorithm === "random") return { ...config, weights: null, seed: null, fallback_enabled: true };
  return {
    ...config,
    fallback_enabled: true,
    [algorithm]: { ...switchyardRecord(config[algorithm]), ...SWITCHYARD_DEFAULTS[algorithm] },
  };
};

export const hydrateSwitchyardConfig = (value: unknown): SwitchyardDraft => switchyardRecord(jsonValue(value));
export const switchyardConfigError = (config: SwitchyardDraft): string | null => {
  const result = switchyardConfigSchema.safeParse(config);
  return result.success
    ? null
    : result.error.issues.map((issue) => `${issue.path.join(".")}: ${issue.message}`).join(". ");
};
