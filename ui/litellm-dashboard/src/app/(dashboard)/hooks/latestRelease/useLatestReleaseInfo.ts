import { useQuery } from "@tanstack/react-query";
import { backgroundFetchClient } from "@/lib/http/api";
import type { components } from "@/lib/http/schema";
import { createQueryKeys } from "../common/queryKeysFactory";

export type LatestReleaseInfo = components["schemas"]["LatestReleaseInfo"];

export type LatestReleaseAnswer = { release: LatestReleaseInfo | null; pending: boolean };

export const PENDING_RELEASE_POLL_MS = 5 * 1000;

export const latestReleaseKeys = createQueryKeys("latestRelease");

export const pendingReleasePollMs = (state: { status: string; data?: LatestReleaseAnswer }) =>
  state.status === "success" && state.data?.pending ? PENDING_RELEASE_POLL_MS : false;

export const keepKnownRelease = (known: unknown, answer: unknown) => {
  const previous = known as LatestReleaseAnswer | undefined;
  const next = answer as LatestReleaseAnswer;
  return next.pending && previous?.release ? { ...next, release: previous.release } : next;
};

export const fetchLatestRelease = async (): Promise<LatestReleaseAnswer> => {
  const { data, response } = await backgroundFetchClient.GET("/get/latest_release_info");
  return { release: data ?? null, pending: response.status === 202 };
};

export const useLatestReleaseInfo = (accessToken: string | null | undefined) => {
  const options = {
    queryKey: latestReleaseKeys.all,
    queryFn: fetchLatestRelease,
    enabled: Boolean(accessToken),
    staleTime: 60 * 60 * 1000,
    retry: false,
    refetchInterval: (query: { state: { status: string; data?: LatestReleaseAnswer } }) =>
      pendingReleasePollMs(query.state),
    structuralSharing: keepKnownRelease,
  };
  const { data, isError } = useQuery(options);
  return { data: isError ? undefined : data?.release };
};
