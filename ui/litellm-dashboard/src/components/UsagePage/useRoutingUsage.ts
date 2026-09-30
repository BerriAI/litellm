import { $api } from "@/lib/http/api";
import type { RoutingUsageQuery } from "./routingUsage";

export const useRoutingUsage = (query: RoutingUsageQuery, enabled = true) =>
  $api.useQuery("get", "/auto_router/usage", { params: { query } }, { enabled, retry: false, staleTime: 60_000 });
