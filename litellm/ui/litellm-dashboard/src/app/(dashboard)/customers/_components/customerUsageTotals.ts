import type { DailyData } from "@/components/UsagePage/types";
import type { components } from "@/lib/http/schema";

type Totals = Readonly<{ spend: number; requests: number; tokens: number }>;

export function customerUsageTotals(
  days: readonly (components["schemas"]["DailySpendData"] | DailyData)[],
): Record<string, Totals> {
  return days.reduce<Record<string, Totals>>((totals, day) => {
    for (const [id, entity] of Object.entries(day.breakdown?.entities ?? {})) {
      const previous = totals[id];
      totals[id] = {
        spend: (previous?.spend ?? 0) + entity.metrics.spend,
        requests: (previous?.requests ?? 0) + entity.metrics.api_requests,
        tokens: (previous?.tokens ?? 0) + entity.metrics.total_tokens,
      };
    }
    return totals;
  }, {});
}
