import { useMutation, useQueryClient, type QueryClient } from "@tanstack/react-query";

import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { $api, fetchClient } from "@/lib/http/api";
import { toast } from "@/lib/toast";
import { isDiscoveryBusy, isoDay, type DraftTaskEdit } from "./taskInsightsData";

const OVERVIEW = "/task-insights/overview" as const;
const MATRIX = "/task-insights/matrix" as const;
const TASK = "/task-insights/tasks/{task_id}" as const;
const TAXONOMY = "/task-insights/taxonomy" as const;
const BUSY_POLL_MS = 10_000;

export interface TaskInsightsFilter {
  startTime: Date | null;
  endTime: Date | null;
  teamId: string | null;
}

const windowQuery = ({ startTime, endTime, teamId }: TaskInsightsFilter) => ({
  start_date: startTime ? isoDay(startTime) : undefined,
  end_date: endTime ? isoDay(endTime) : undefined,
  team_id: teamId ?? undefined,
});

export const useTaskInsightsOverview = (filter: TaskInsightsFilter, enabled: boolean) => {
  const { accessToken } = useAuthorized();
  return $api.useQuery(
    "get",
    OVERVIEW,
    { params: { query: windowQuery(filter) } },
    {
      enabled: enabled && Boolean(accessToken),
      retry: 1,
      refetchInterval: (query) => (isDiscoveryBusy(query.state.data?.status) ? BUSY_POLL_MS : false),
    },
  );
};

export const useTaskInsightsMatrix = (filter: TaskInsightsFilter, enabled: boolean) => {
  const { accessToken } = useAuthorized();
  return $api.useQuery(
    "get",
    MATRIX,
    { params: { query: windowQuery(filter) } },
    { enabled: enabled && Boolean(accessToken), retry: 1 },
  );
};

export const useTaskInsightsTask = (filter: TaskInsightsFilter, taskId: string | null, modelConfigId: string | null) => {
  const { accessToken } = useAuthorized();
  return $api.useQuery(
    "get",
    TASK,
    {
      params: {
        path: { task_id: taskId ?? "" },
        query: { ...windowQuery(filter), model_config_id: modelConfigId ?? undefined },
      },
    },
    { enabled: Boolean(accessToken) && taskId !== null, retry: 1 },
  );
};

export const useTaskInsightsTaxonomy = (enabled: boolean) => {
  const { accessToken } = useAuthorized();
  return $api.useQuery("get", TAXONOMY, {}, { enabled: enabled && Boolean(accessToken), retry: 1 });
};

const invalidateAll = (queryClient: QueryClient) =>
  Promise.all(
    [OVERVIEW, MATRIX, TASK, TAXONOMY].map((path) => queryClient.invalidateQueries({ queryKey: ["get", path] })),
  );

const useTaskInsightsMutation = <TVariables, TResult>(mutationFn: (variables: TVariables) => Promise<TResult>) => {
  const queryClient = useQueryClient();
  return useMutation({
    mutationFn,
    onSuccess: () => invalidateAll(queryClient),
    onError: (error: unknown) => toast.fromError(error),
  });
};

export const useRequestDiscovery = () =>
  useTaskInsightsMutation(async (filter: TaskInsightsFilter) => {
    const { start_date, end_date } = windowQuery(filter);
    const { data } = await fetchClient.POST("/task-insights/discovery", { body: { start_date, end_date } });
    return data;
  });

export const useEditDraftTask = () =>
  useTaskInsightsMutation(async ({ taskId, edit }: { taskId: string; edit: DraftTaskEdit }) => {
    const { data } = await fetchClient.PATCH(TASK, { params: { path: { task_id: taskId } }, body: edit });
    return data;
  });

export const useFreezeTaxonomy = () =>
  useTaskInsightsMutation(async (draftTaskCount: number) => {
    const { data } = await fetchClient.POST("/task-insights/taxonomy/freeze", {
      body: { draft_task_count: draftTaskCount },
    });
    return data;
  });
