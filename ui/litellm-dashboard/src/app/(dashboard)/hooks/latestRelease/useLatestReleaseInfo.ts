import { $api } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";

export type LatestReleaseInfo = components["schemas"]["LatestReleaseInfo"];

export const useLatestReleaseInfo = (accessToken: string | null | undefined) =>
  $api.useQuery(
    "get",
    "/get/latest_release_info",
    {},
    {
      enabled: Boolean(accessToken),
      staleTime: 60 * 60 * 1000,
      retry: false,
    },
  );
