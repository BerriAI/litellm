import { $api } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";

export type LatestReleaseInfo = components["schemas"]["LatestReleaseInfo"];

export const PENDING_RELEASE_POLL_MS = 5 * 1000;
export const PENDING_RELEASE_ANSWERS_PER_BURST = 3;

type ReleaseQueryState = { status: string; data: unknown; dataUpdateCount: number };

export const pendingReleasePollMs = ({ status, data, dataUpdateCount }: ReleaseQueryState) =>
  status === "success" && data === null && dataUpdateCount % PENDING_RELEASE_ANSWERS_PER_BURST !== 0
    ? PENDING_RELEASE_POLL_MS
    : false;

export const keepKnownRelease = (known: unknown, answer: unknown) =>
  answer === null && known !== undefined ? known : answer;

export const useLatestReleaseInfo = (accessToken: string | null | undefined) => {
  const options = {
    enabled: Boolean(accessToken),
    staleTime: 60 * 60 * 1000,
    retry: false,
    refetchInterval: (query: { state: ReleaseQueryState }) => pendingReleasePollMs(query.state),
    structuralSharing: keepKnownRelease,
  };
  return $api.useQuery("get", "/get/latest_release_info", {}, options);
};
