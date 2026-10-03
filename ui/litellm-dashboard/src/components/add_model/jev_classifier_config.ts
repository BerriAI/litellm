import { z } from "zod";
import type { ClassifierType } from "./classifier_types";

export const OSS_CLASSIFIER_MODELS = {
  laya: ["english", "multilingual", "typed-decisions"],
  bespoke: ["nimble-latest", "nimble", "bespokelabs/Bespoke-Nimble-9B"],
} as const;

const jevClassifierConfigFields = {
  provider: z.preprocess(
    (value) => (value === "typesafe" ? "jev" : value),
    z.enum(["jev", "laya", "bespoke"]).optional(),
  ),
  model: z.string().trim().min(1).optional(),
  timeout_ms: z.number().int().positive().default(3000),
  instructions: z
    .string()
    .nullish()
    .transform((value) => value ?? undefined),
  circuit_breaker_enabled: z.boolean().optional(),
  circuit_breaker_cooldown_seconds: z.number().finite().positive().optional(),
};

export const jevClassifierConfigSchema = z
  .object(jevClassifierConfigFields)
  .transform((config) => ({
    ...config,
    model:
      config.model ??
      (config.provider && config.provider !== "jev" ? OSS_CLASSIFIER_MODELS[config.provider][0] : "jev-latest"),
  }))
  .refine(
    (config) =>
      !config.provider ||
      config.provider === "jev" ||
      OSS_CLASSIFIER_MODELS[config.provider].some((model) => model === config.model),
    { message: "Select a supported classifier model", path: ["model"] },
  );

export type JevClassifierConfig = z.infer<typeof jevClassifierConfigSchema>;

export const defaultJevClassifierConfig = (provider: JevClassifierConfig["provider"] = "jev"): JevClassifierConfig =>
  jevClassifierConfigSchema.parse({ provider });

export const hydrateOssClassifier = (config: {
  classifier_type?: ClassifierType | "oss_classifier";
  opensource_classifier_config?: unknown;
  jev_classifier_config?: unknown;
}): { classifier_type: ClassifierType; jev_classifier_config?: JevClassifierConfig } => ({
  classifier_type: config.classifier_type === "oss_classifier" ? "jev" : config.classifier_type ?? "heuristic",
  jev_classifier_config:
    config.classifier_type === "oss_classifier" || config.classifier_type === "jev"
      ? jevClassifierConfigSchema.safeParse(config.opensource_classifier_config ?? config.jev_classifier_config ?? {})
          .data ?? defaultJevClassifierConfig()
      : undefined,
});

export const normalizeJevClassifierConfig = (
  config: JevClassifierConfig = defaultJevClassifierConfig(),
): JevClassifierConfig => ({
  provider: config.provider ?? "jev",
  model: config.model.trim(),
  timeout_ms: config.timeout_ms,
  ...(config.instructions?.trim() && { instructions: config.instructions.trim() }),
  ...(config.circuit_breaker_enabled !== undefined && { circuit_breaker_enabled: config.circuit_breaker_enabled }),
  ...(config.circuit_breaker_cooldown_seconds !== undefined && {
    circuit_breaker_cooldown_seconds: config.circuit_breaker_cooldown_seconds,
  }),
});
