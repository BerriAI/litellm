import { useQuery } from "@tanstack/react-query";
import { dailyActivityAggregatedCall } from "@/components/networking";
import { toDailyData } from "@/components/UsagePage/dailyActivityApi";
import type { DailyData, SpendMetrics } from "@/components/UsagePage/types";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";

export type CustomerStatus = "active" | "over_budget" | "blocked";

export const monthRange = (monthsBack = 0, now = new Date()): { start: Date; end: Date } => ({
  start: new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() - monthsBack, 1)),
  end: new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() - monthsBack + 1, 0, 23, 59, 59)),
});

export interface Totals {
  spend: number;
  requests: number;
  inputTokens: number;
  outputTokens: number;
}

export interface CustomerKeyUsage extends Totals {
  hash: string;
  alias: string | null;
}

export interface CustomerUsage extends Totals {
  keys: readonly CustomerKeyUsage[];
}

const ZERO: Totals = { spend: 0, requests: 0, inputTokens: 0, outputTokens: 0 };

export const addMetrics = (t: Totals, m: Partial<SpendMetrics> | undefined): Totals => ({
  spend: t.spend + (m?.spend ?? 0),
  requests: t.requests + (m?.api_requests ?? 0),
  inputTokens: t.inputTokens + (m?.prompt_tokens ?? 0),
  outputTokens: t.outputTokens + (m?.completion_tokens ?? 0),
});

export const usageByCustomer = (days: readonly DailyData[]): Map<string, CustomerUsage> => {
  const totals = new Map<string, Totals>();
  const keys = new Map<string, Map<string, CustomerKeyUsage>>();
  for (const day of days) {
    for (const [id, entity] of Object.entries(day.breakdown.entities)) {
      totals.set(id, addMetrics(totals.get(id) ?? ZERO, entity.metrics));
      const byKey = keys.get(id) ?? new Map<string, CustomerKeyUsage>();
      for (const [hash, key] of Object.entries(entity.api_key_breakdown)) {
        const prev = byKey.get(hash) ?? { ...ZERO, hash, alias: key.metadata.key_alias };
        byKey.set(hash, { ...prev, ...addMetrics(prev, key.metrics), alias: prev.alias ?? key.metadata.key_alias });
      }
      keys.set(id, byKey);
    }
  }
  return new Map(
    [...totals.entries()].map(([id, t]) => [
      id,
      { ...t, keys: [...(keys.get(id)?.values() ?? [])].sort((a, b) => b.spend - a.spend) },
    ]),
  );
};

export const useCustomerUsage = (accessToken: string | null, monthsBack = 0) =>
  useQuery({
    queryKey: ["customer-portal", "usage", monthsBack],
    enabled: Boolean(accessToken),
    queryFn: async () => {
      const { start, end } = monthRange(monthsBack);
      const res = await dailyActivityAggregatedCall("customer", {
        accessToken: accessToken!,
        startTime: start,
        endTime: end,
      });
      return usageByCustomer(toDailyData(res));
    },
  });

export const customerStatus = (c: Pick<EndUser, "blocked" | "litellm_budget_table">, spend: number): CustomerStatus => {
  if (c.blocked) return "blocked";
  const max = c.litellm_budget_table?.max_budget;
  return max !== null && max !== undefined && spend >= max ? "over_budget" : "active";
};

export const rankBySpend = (customers: readonly EndUser[], usage: ReadonlyMap<string, CustomerUsage> | undefined) =>
  customers
    .map((c) => {
      const u = usage?.get(c.user_id);
      const spend = u?.spend ?? 0;
      return { c, spend, keys: u?.keys.length ?? 0, status: customerStatus(c, spend) };
    })
    .sort((a, b) => b.spend - a.spend);
