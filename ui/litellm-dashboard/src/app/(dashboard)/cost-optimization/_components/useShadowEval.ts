import { useQueryClient } from "@tanstack/react-query";

import { $api } from "@/lib/http/api";

import type { components } from "@/lib/http/schema";

export type ShadowEvalJob = components["schemas"]["GetShadowEvalJobResponse"];
export type ShadowEvalTierResult = components["schemas"]["ShadowEvalTierResult"];
export type StartShadowEvalRequest = components["schemas"]["StartShadowEvalRequest"];

const JOBS_PATH = "/auto_router/shadow_eval" as const;

/** Poll faster while a job is actively collecting verdicts. */
const ACTIVE_POLL_MS = 15_000;

export const useShadowEvalJobs = (accessToken: string | null) =>
  $api.useQuery("get", JOBS_PATH, {}, { enabled: Boolean(accessToken), retry: false });

export const useShadowEvalJob = (accessToken: string | null, jobId: string | null) =>
  $api.useQuery(
    "get",
    "/auto_router/shadow_eval/{job_id}",
    { params: { path: { job_id: jobId ?? "" } } },
    {
      enabled: Boolean(accessToken) && Boolean(jobId),
      retry: false,
      refetchInterval: (query) => {
        const status = query.state.data?.status;
        return status === "pending" || status === "running" ? ACTIVE_POLL_MS : false;
      },
    },
  );

export const useStartShadowEval = () => {
  const queryClient = useQueryClient();
  return $api.useMutation("post", "/auto_router/shadow_eval/start", {
    onSuccess: () => queryClient.invalidateQueries(),
  });
};

export const useStopShadowEval = () => {
  const queryClient = useQueryClient();
  return $api.useMutation("post", "/auto_router/shadow_eval/{job_id}/stop", {
    onSuccess: () => queryClient.invalidateQueries(),
  });
};
