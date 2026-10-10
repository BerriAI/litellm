import { z } from "zod";
import type { ClassifierType } from "./classifier_types";

export const OSS_CLASSIFIER_MODELS = {
  laya: ["english", "multilingual", "typed-decisions"],
  bespoke: ["nimble-latest", "nimble", "bespokelabs/Bespoke-Nimble-9B"],
} as const;

const OSS_CLASSIFIER_PROVIDERS = ["jev", "laya", "bespoke", "databricks"] as const;
export type OssClassifierProvider = (typeof OSS_CLASSIFIER_PROVIDERS)[number];
export const isOssClassifierProvider = (value: unknown): value is OssClassifierProvider =>
  OSS_CLASSIFIER_PROVIDERS.some((provider) => provider === value);

const DEFAULT_TIMEOUT_MS = 3000;
const SERVING_ENDPOINT_NAME = /^[A-Za-z0-9_-][A-Za-z0-9._-]*$/;
const DATABRICKS_AI_DECIDE_MODEL = "ai_decide";

export const fixedClassifierModels = (provider: OssClassifierProvider | undefined): readonly string[] | undefined =>
  provider === "laya" || provider === "bespoke" ? OSS_CLASSIFIER_MODELS[provider] : undefined;

const defaultClassifierModel = (provider: OssClassifierProvider | undefined): string =>
  fixedClassifierModels(provider)?.[0] ?? (provider === "databricks" ? DATABRICKS_AI_DECIDE_MODEL : "jev-latest");

const isSupportedClassifierModel = (provider: OssClassifierProvider | undefined, model: string): boolean => {
  if (provider === "databricks") return SERVING_ENDPOINT_NAME.test(model);
  const fixed = fixedClassifierModels(provider);
  return fixed === undefined || fixed.some((candidate) => candidate === model);
};

const jevClassifierConfigFields = {
  provider: z.preprocess(
    (value) => (value === "typesafe" ? "jev" : value),
    z.enum(["jev", "laya", "bespoke", "databricks"]).optional(),
  ),
  model: z.string().trim().min(1).optional(),
  timeout_ms: z.number().int().positive().default(DEFAULT_TIMEOUT_MS),
  instructions: z
    .string()
    .nullish()
    .transform((value) => value ?? undefined)
    .optional(),
  circuit_breaker_enabled: z.boolean().optional(),
  circuit_breaker_cooldown_seconds: z.number().finite().positive().optional(),
};

export const jevClassifierConfigSchema = z
  .object(jevClassifierConfigFields)
  .transform((config) => ({ ...config, model: config.model ?? defaultClassifierModel(config.provider) }))
  .refine((config) => isSupportedClassifierModel(config.provider, config.model), {
    error: "Select a supported classifier model, or enter ai_decide or a bare Databricks serving endpoint name",
    path: ["model"],
  });

export type JevClassifierConfig = z.infer<typeof jevClassifierConfigSchema>;

export const defaultJevClassifierConfig = (provider: OssClassifierProvider = "jev"): JevClassifierConfig =>
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
