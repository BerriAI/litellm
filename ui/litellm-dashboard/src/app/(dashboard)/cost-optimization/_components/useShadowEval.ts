import { useMutation, useQueryClient, type QueryClient } from "@tanstack/react-query";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { toast } from "@/lib/toast";
import { $api, fetchClient } from "@/lib/http/api";

import type { components } from "@/lib/http/schema";

export type ShadowEvalJob = components["schemas"]["ShadowEvalJobResponse"];
export type ShadowEvalJobKey = components["schemas"]["ShadowEvalJobKeyResponse"];
export type ShadowEvalSlice = components["schemas"]["ShadowEvalSlice"];
export type StartShadowEvalRequest = components["schemas"]["StartShadowEvalRequest"];

const LIST_PATH = "/auto_router/shadow_eval" as const;
const DETAIL_PATH = "/auto_router/shadow_eval/{job_id}" as const;

const ACTIVE_POLL_MS = 15_000;

export const shadowEvalPollMs = (status: ShadowEvalJob["status"] | undefined): number | false =>
  status === "running" || status === undefined ? ACTIVE_POLL_MS : false;

export const shadowEvalListPollMs = (jobs: ShadowEvalJob[] | undefined): number | false =>
  jobs?.some((job) => job.status === "running") ? ACTIVE_POLL_MS : false;

const invalidateShadowEval = (queryClient: QueryClient) =>
  Promise.all([
    queryClient.invalidateQueries({ queryKey: ["get", LIST_PATH] }),
    queryClient.invalidateQueries({ queryKey: ["get", DETAIL_PATH] }),
  ]);

export const useShadowEvalJobs = () => {
  const { accessToken } = useAuthorized();
  return $api.useQuery(
    "get",
    LIST_PATH,
    {},
    {
      enabled: Boolean(accessToken),
      retry: 1,
      refetchInterval: (query) => shadowEvalListPollMs(query.state.data),
    },
  );
};

export const useShadowEvalJob = (jobId: string | null) => {
  const { accessToken } = useAuthorized();
  return $api.useQuery(
    "get",
    DETAIL_PATH,
    { params: { path: { job_id: jobId ?? "" } } },
    {
      enabled: Boolean(accessToken) && Boolean(jobId),
      retry: 1,
      refetchInterval: (query) => shadowEvalPollMs(query.state.data?.status),
    },
  );
};

const useShadowEvalMutation = <TVariables>(mutationFn: (variables: TVariables) => Promise<unknown>) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn,
    onSuccess: () => invalidateShadowEval(queryClient),
    onError: (error: unknown) => toast.fromError(error),
  });
};

export const useStartShadowEval = () =>
  useShadowEvalMutation(async (body: StartShadowEvalRequest) => {
    const { data } = await fetchClient.POST("/auto_router/shadow_eval/start", { body });
    return data;
  });

export const useStopShadowEval = () =>
  useShadowEvalMutation(async (jobId: string) => {
    const { data } = await fetchClient.POST("/auto_router/shadow_eval/{job_id}/stop", {
      params: { path: { job_id: jobId } },
    });
    return data;
  });
