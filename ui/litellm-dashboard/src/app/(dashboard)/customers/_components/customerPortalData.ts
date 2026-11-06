import { useQuery } from "@tanstack/react-query";
import { dailyActivityAggregatedCall } from "@/components/networking";
import type { EndUser } from "@/app/(dashboard)/hooks/customers/useCustomers";

export type CustomerStatus = "active" | "over_budget" | "blocked";

export const monthRange = (monthsBack = 0): { start: Date; end: Date } => {
  const now = new Date();
  const start = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() - monthsBack, 1));
  const end = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth() - monthsBack + 1, 0, 23, 59, 59));
  return { start, end };
};

export interface CustomerSpend {
  spend: number;
  requests: number;
  keys: ReadonlySet<string>;
}

export const useCustomerSpendByCustomer = (accessToken: string | null, monthsBack = 0) =>
  useQuery({
    queryKey: ["customer-portal", "spend", monthsBack],
    enabled: Boolean(accessToken),
    queryFn: async () => {
      const { start, end } = monthRange(monthsBack);
      const res = await dailyActivityAggregatedCall("customer", { accessToken: accessToken!, startTime: start, endTime: end });
      const out = new Map<string, { spend: number; requests: number; keys: Set<string> }>();
      for (const day of res.results) {
        const entities = (day.breakdown?.entities ?? {}) as Record<string, { metrics?: { spend?: number; api_requests?: number } }>;
        for (const [id, e] of Object.entries(entities)) {
          const cur = out.get(id) ?? { spend: 0, requests: 0, keys: new Set<string>() };
          cur.spend += e.metrics?.spend ?? 0;
          cur.requests += e.metrics?.api_requests ?? 0;
          out.set(id, cur);
        }
      }
      await Promise.all(
        [...out.entries()]
          .filter(([, v]) => v.requests > 0)
          .map(async ([id, v]) => {
            const per = await dailyActivityAggregatedCall("customer", { accessToken: accessToken!, startTime: start, endTime: end, entityIds: [id] });
            for (const day of per.results) for (const hash of Object.keys(day.breakdown?.api_keys ?? {})) v.keys.add(hash);
          }),
      );
      return out as Map<string, CustomerSpend>;
    },
  });

export const customerStatus = (c: EndUser, spend: number): CustomerStatus => {
  if (c.blocked) return "blocked";
  const max = c.litellm_budget_table?.max_budget ?? null;
  if (max !== null && max !== undefined && spend >= max) return "over_budget";
  return "active";
};

export const formatUSD = (n: number) => n.toLocaleString("en-US", { style: "currency", currency: "USD", maximumFractionDigits: 2 });
