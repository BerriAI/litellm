import type { components, paths } from "@/lib/http/schema";

export type RoutingUsage = components["schemas"]["AutoRouterUsage"];
export type RoutingUsageQuery = paths["/auto_router/usage"]["get"]["parameters"]["query"];
export type RoutingUsageScope = Pick<RoutingUsageQuery, "start_date" | "end_date" | "user_id" | "api_key">;

export const routingSources = (rows: readonly RoutingUsage[]) => {
  const names = [null, ...new Set(rows.flatMap((row) => (row.router_name === null ? [] : [row.router_name])))];
  return names.map((name) => ({
    name,
    requests: rows.filter((row) => row.router_name === name).reduce((sum, row) => sum + row.requests, 0),
    spend: rows.filter((row) => row.router_name === name).reduce((sum, row) => sum + row.spend, 0),
  }));
};

export const routingTiers = (rows: readonly RoutingUsage[]) =>
  [...new Set(rows.map((row) => row.tier))].map((tier) => {
    const tierRows = rows.filter((row) => row.tier === tier);
    return {
      tier,
      requests: tierRows.reduce((sum, row) => sum + row.requests, 0),
      spend: tierRows.reduce((sum, row) => sum + row.spend, 0),
      models: [...new Set(tierRows.map((row) => row.model))].map((model) => ({
        model,
        requests: tierRows.filter((row) => row.model === model).reduce((sum, row) => sum + row.requests, 0),
        spend: tierRows.filter((row) => row.model === model).reduce((sum, row) => sum + row.spend, 0),
      })),
    };
  });

export const trafficShare = (requests: number, total: number) =>
  `${(total > 0 ? (100 * requests) / total : 0).toLocaleString(undefined, { maximumFractionDigits: 1 })}%`;

export const routingSpend = (spend: number) =>
  spend.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: spend > 0 && spend < 0.01 ? 6 : 2,
  });
