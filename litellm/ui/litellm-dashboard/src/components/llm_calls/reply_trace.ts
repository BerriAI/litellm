export type GatewayTrace = Readonly<{
  callId?: string;
  modelGroup?: string;
  deploymentId?: string;
  providerModel?: string;
  apiBase?: string;
  attemptedFallbacks?: number;
  attemptedRetries?: number;
  overheadMs?: number;
  responseMs?: number;
  cost?: number;
  appliedGuardrails: readonly string[];
  appliedPolicies?: readonly string[];
  cacheHit: boolean;
}>;

const text = (headers: Headers, name: string): string | undefined => {
  const value = headers.get(name)?.trim();
  return value ? value : undefined;
};

const number = (headers: Headers, name: string): number | undefined => {
  const value = text(headers, name);
  if (value === undefined) return undefined;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : undefined;
};

const list = (headers: Headers, name: string): readonly string[] =>
  (text(headers, name) ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item !== "");

export const gatewayTraceFromHeaders = (headers: Headers): GatewayTrace => ({
  callId: text(headers, "x-litellm-call-id"),
  modelGroup: text(headers, "x-litellm-model-group"),
  deploymentId: text(headers, "x-litellm-model-id"),
  providerModel: text(headers, "x-litellm-model-name"),
  apiBase: text(headers, "x-litellm-model-api-base"),
  attemptedFallbacks: number(headers, "x-litellm-attempted-fallbacks"),
  attemptedRetries: number(headers, "x-litellm-attempted-retries"),
  overheadMs: number(headers, "x-litellm-overhead-duration-ms"),
  responseMs: number(headers, "x-litellm-response-duration-ms"),
  cost: number(headers, "x-litellm-response-cost"),
  appliedGuardrails: list(headers, "x-litellm-applied-guardrails"),
  appliedPolicies: headers.has("x-litellm-call-id") ? list(headers, "x-litellm-applied-policies") : undefined,
  cacheHit: text(headers, "x-litellm-cache-key") !== undefined,
});
