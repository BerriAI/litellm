import { $api } from "@/lib/http/api";

import { windowFor, type BenchmarkWindow } from "./autoRouterBenchmarks";

export const useAutoRouterBenchmarks = (accessToken: string | null, range: BenchmarkWindow) =>
  $api.useQuery(
    "get",
    "/auto_router/benchmarks",
    { params: { query: windowFor(range, new Date()) } },
    { enabled: Boolean(accessToken), retry: false },
  );
