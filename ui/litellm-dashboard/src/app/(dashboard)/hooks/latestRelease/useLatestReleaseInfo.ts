import { $api } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";

export type LatestReleaseInfo = components["schemas"]["LatestReleaseInfo"];

export const PENDING_RELEASE_POLL_MS = 5 * 1000;
export const PENDING_RELEASE_MAX_ANSWERS = 3;

export const pendingReleasePollMs = (state: { status: string; data: unknown; dataUpdateCount: number }) =>
  state.status === "success" && state.data === null && state.dataUpdateCount < PENDING_RELEASE_MAX_ANSWERS
    ? PENDING_RELEASE_POLL_MS
    : false;

export const useLatestReleaseInfo = (accessToken: string | null | undefined) => {
  const options = {
    enabled: Boolean(accessToken),
    staleTime: 60 * 60 * 1000,
    retry: false,
    refetchInterval: (query: { state: Parameters<typeof pendingReleasePollMs>[0] }) =>
      pendingReleasePollMs(query.state),
  };
  return $api.useQuery("get", "/get/latest_release_info", {}, options);
};
