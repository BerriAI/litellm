import { z } from "zod";

const jevClassifierConfigFields = {
  provider: z.enum(["typesafe", "laya"]).optional(),
  model: z.string().trim().min(1).optional(),
  timeout_ms: z.number().int().positive().default(3000),
  instructions: z
    .string()
    .nullish()
    .transform((value) => value ?? undefined),
  circuit_breaker_enabled: z.boolean().optional(),
  circuit_breaker_cooldown_seconds: z.number().finite().positive().optional(),
};

export const jevClassifierConfigSchema = z.object(jevClassifierConfigFields).transform((config) => ({
  ...config,
  model: config.model ?? (config.provider === "laya" ? "multilingual" : "jev-latest"),
}));

export type JevClassifierConfig = z.infer<typeof jevClassifierConfigSchema>;

export const defaultJevClassifierConfig = (): JevClassifierConfig => jevClassifierConfigSchema.parse({});

export const normalizeJevClassifierConfig = (
  config: JevClassifierConfig = defaultJevClassifierConfig(),
): JevClassifierConfig => ({
  ...(config.provider && { provider: config.provider }),
  model: config.model.trim(),
  timeout_ms: config.timeout_ms,
  ...(config.instructions?.trim() && { instructions: config.instructions.trim() }),
  ...(config.circuit_breaker_enabled !== undefined && { circuit_breaker_enabled: config.circuit_breaker_enabled }),
  ...(config.circuit_breaker_cooldown_seconds !== undefined && {
    circuit_breaker_cooldown_seconds: config.circuit_breaker_cooldown_seconds,
  }),
});
