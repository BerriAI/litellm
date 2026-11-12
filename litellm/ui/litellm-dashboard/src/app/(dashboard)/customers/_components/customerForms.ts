import type { components } from "@/lib/http/schema";

export type NewCustomerBody = components["schemas"]["NewCustomerRequest"];
export type UpdateCustomerBody = components["schemas"]["UpdateCustomerRequest"];
export type CustomerBudgetBody = components["schemas"]["BudgetNewRequest"];

export const digitsOnly = (s: string) => s.replace(/[^0-9]/g, "");
export const decimalOnly = (s: string) => s.replace(/[^0-9.]/g, "");

const numberOrNull = (s: string): number | null => (s.trim() === "" ? null : Number(s));

type ParsedRateLimit = { valid: true; value: number | null } | { valid: false };
const rateLimitScales: Readonly<Record<string, number>> = { k: 3, m: 6, b: 9 };

export const parseRateLimit = (text: string): ParsedRateLimit => {
  const value = text.trim();
  if (!value) return { valid: true, value: null };
  if (value.length > 48) return { valid: false };
  const match = /^((?:\d+|\d{1,3}(?:,\d{3})+)(?:\.\d+)?|\.\d+)\s*([kmb])?$/i.exec(value);
  if (!match) return { valid: false };
  const [whole, fraction = ""] = match[1].replaceAll(",", "").split(".");
  const scale = rateLimitScales[match[2]?.toLowerCase() ?? ""] ?? 0;
  if (/[^0]/.test(fraction.slice(scale))) return { valid: false };
  const amount = Number((whole || "0") + fraction.slice(0, scale).padEnd(scale, "0"));
  return Number.isSafeInteger(amount) ? { valid: true, value: amount } : { valid: false };
};

export const formatRateLimit = (text: string): string => {
  const parsed = parseRateLimit(text);
  if (!parsed.valid) return text;
  return parsed.value === null ? "" : new Intl.NumberFormat("en-US").format(parsed.value);
};

const rateLimitNumber = (text: string): number | null => {
  const parsed = parseRateLimit(text);
  if (!parsed.valid) throw new Error("Rate limits must be non-negative whole numbers, optionally using K, M or B.");
  return parsed.value;
};

export interface ModelRateLimitFields {
  model: string;
  tpm: string;
  rpm: string;
}

export interface CustomerFallbackFields {
  model: string;
  fallbacks: string[];
}

export const customerFallbackFields = (
  fallbacks: readonly Record<string, readonly string[]>[] | null | undefined,
): CustomerFallbackFields[] | null =>
  fallbacks == null
    ? null
    : fallbacks.flatMap((chain) =>
        Object.entries(chain).map(([model, targets]) => ({ model, fallbacks: [...targets] })),
      );

export const buildCustomerFallbacks = (
  rules: readonly CustomerFallbackFields[] | null,
): Record<string, string[]>[] | null =>
  rules === null ? null : rules.map((rule) => ({ [rule.model]: rule.fallbacks }));

export const hasIncompleteCustomerFallbacks = (rules: readonly CustomerFallbackFields[] | null): boolean =>
  rules?.some((rule) => !rule.model.trim() || rule.fallbacks.length === 0) ?? false;

export interface CustomerRateLimitFields {
  tpm: string;
  rpm: string;
  modelLimits?: readonly ModelRateLimitFields[];
}

export const hasInvalidCustomerRateLimits = (fields: CustomerRateLimitFields): boolean =>
  [fields.tpm, fields.rpm, ...(fields.modelLimits ?? []).flatMap((row) => [row.tpm, row.rpm])].some(
    (text) => !parseRateLimit(text).valid,
  );

export const modelRateLimitFields = (
  budget: Pick<CustomerBudgetBody, "model_tpm_limit" | "model_rpm_limit"> | null | undefined,
): ModelRateLimitFields[] => {
  const tokens = budget?.model_tpm_limit ?? {};
  const requests = budget?.model_rpm_limit ?? {};
  return Array.from(new Set([...Object.keys(tokens), ...Object.keys(requests)]))
    .sort()
    .map((model) => ({
      model,
      tpm: tokens[model] === undefined ? "" : String(tokens[model]),
      rpm: requests[model] === undefined ? "" : String(requests[model]),
    }));
};

export const buildCustomerBudgetBody = (fields: CustomerRateLimitFields & { budget: string }): CustomerBudgetBody => {
  const tokens: Record<string, number> = {};
  const requests: Record<string, number> = {};
  for (const row of fields.modelLimits ?? []) {
    const model = row.model.trim();
    if (!model) continue;
    const tpm = rateLimitNumber(row.tpm);
    const rpm = rateLimitNumber(row.rpm);
    if (tpm !== null) tokens[model] = tpm;
    if (rpm !== null) requests[model] = rpm;
  }
  return {
    max_budget: numberOrNull(fields.budget),
    tpm_limit: rateLimitNumber(fields.tpm),
    rpm_limit: rateLimitNumber(fields.rpm),
    model_tpm_limit: tokens,
    model_rpm_limit: requests,
  };
};

export interface CreateCustomerFields extends CustomerRateLimitFields {
  userId: string;
  alias: string;
  budget: string;
  fallbacks?: CustomerFallbackFields[] | null;
}

export const buildCreateCustomerBody = (f: CreateCustomerFields): NewCustomerBody | null => {
  const userId = f.userId.trim();
  if (!userId || hasInvalidCustomerRateLimits(f)) return null;
  const maxBudget = numberOrNull(f.budget);
  const limits = buildCustomerBudgetBody(f);
  return {
    user_id: userId,
    alias: f.alias.trim() || null,
    blocked: false,
    max_budget: maxBudget,
    budget_duration: maxBudget === null ? null : "1mo",
    tpm_limit: limits.tpm_limit,
    rpm_limit: limits.rpm_limit,
    fallbacks: buildCustomerFallbacks(f.fallbacks ?? null),
    ...(Object.keys(limits.model_tpm_limit ?? {}).length > 0 && { model_tpm_limit: limits.model_tpm_limit }),
    ...(Object.keys(limits.model_rpm_limit ?? {}).length > 0 && { model_rpm_limit: limits.model_rpm_limit }),
  };
};

export interface EditCustomerFields {
  userId: string;
  alias: string;
  blocked: boolean;
  fallbacks?: CustomerFallbackFields[] | null;
}

export const buildUpdateCustomerBody = (f: EditCustomerFields): UpdateCustomerBody => ({
  user_id: f.userId,
  alias: f.alias.trim() || null,
  blocked: f.blocked,
  fallbacks: buildCustomerFallbacks(f.fallbacks ?? null),
});
