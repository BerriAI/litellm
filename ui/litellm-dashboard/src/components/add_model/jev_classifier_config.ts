import { z } from "zod";

const jevClassifierConfigFields = {
  provider: z.enum(["typesafe", "bespoke_nimble"]).optional(),
  model: z.string().trim().min(1).default("jev-latest"),
  timeout_ms: z.number().int().positive().default(3000),
  instructions: z
    .string()
    .nullish()
    .transform((value) => value ?? undefined),
  circuit_breaker_enabled: z.boolean().optional(),
  circuit_breaker_cooldown_seconds: z.number().finite().positive().optional(),
};

export const jevClassifierConfigSchema = z.object(jevClassifierConfigFields);

export const jevClassifierFormConfigSchema = z.object({
  ...jevClassifierConfigFields,
  api_base: z.string().trim().nullish(),
  api_key: z.string().trim().nullish(),
});

export type JevClassifierConfig = z.infer<typeof jevClassifierFormConfigSchema>;
export type DecisionModelProvider = NonNullable<JevClassifierConfig["provider"]>;

export const defaultJevClassifierConfig = (): JevClassifierConfig => jevClassifierConfigSchema.parse({});

export const transitionDecisionModelProvider = (
  config: JevClassifierConfig,
  provider: DecisionModelProvider,
): JevClassifierConfig => {
  if ((config.provider ?? "typesafe") === provider) return config;
  const { api_base, api_key, ...settings } = config;
  return { ...settings, provider, model: provider === "bespoke_nimble" ? "nimble-latest" : "jev-latest" };
};

export const normalizeJevClassifierConfig = (
  config: JevClassifierConfig = defaultJevClassifierConfig(),
): JevClassifierConfig => ({
  ...(config.provider !== undefined && { provider: config.provider }),
  model: config.model.trim(),
  timeout_ms: config.timeout_ms,
  ...(config.api_base === null && { api_base: null }),
  ...(config.api_base?.trim() && { api_base: config.api_base.trim() }),
  ...(config.api_key === null && { api_key: null }),
  ...(config.api_key?.trim() && { api_key: config.api_key.trim() }),
  ...(config.instructions?.trim() && { instructions: config.instructions.trim() }),
  ...(config.circuit_breaker_enabled !== undefined && { circuit_breaker_enabled: config.circuit_breaker_enabled }),
  ...(config.circuit_breaker_cooldown_seconds !== undefined && {
    circuit_breaker_cooldown_seconds: config.circuit_breaker_cooldown_seconds,
  }),
});
