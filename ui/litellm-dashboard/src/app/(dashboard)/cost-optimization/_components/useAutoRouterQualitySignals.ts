import { $api } from "@/lib/http/api";

import { windowFor, type BenchmarkWindow } from "./autoRouterBenchmarks";

export const useAutoRouterQualitySignals = (accessToken: string | null, range: BenchmarkWindow) =>
  $api.useQuery(
    "get",
    "/auto_router/quality_signals",
    { params: { query: windowFor(range, new Date()) } },
    { enabled: Boolean(accessToken), retry: false },
  );
