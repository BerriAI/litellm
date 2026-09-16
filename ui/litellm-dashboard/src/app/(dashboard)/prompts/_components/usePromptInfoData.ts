import { useQuery, type UseQueryOptions } from "@tanstack/react-query";
import { useEffect } from "react";

import { getPromptInfo, getPromptVersions } from "@/components/networking";
import { toast } from "@/lib/toast";

import { usePromptInfoUrlState } from "./usePromptsUrlState";

export type PromptInfoResponse = Awaited<ReturnType<typeof getPromptInfo>>;
type PromptVersionsResponse = Awaited<ReturnType<typeof getPromptVersions>>;
type PromptInfoUrlState = ReturnType<typeof usePromptInfoUrlState>;

const loadPromptInfo = (accessToken: string, promptId: string, version: number | null, environment?: string) =>
  getPromptInfo(accessToken, version === null ? promptId : `${promptId}.v${version}`, environment);

function useDropFailedSelection(isError: boolean, urlState: PromptInfoUrlState): boolean {
  const { version, environment, clearVersion, clearEnvironment } = urlState;
  useEffect(() => {
    if (!isError) return;
    toast.fromError(version === null ? "Failed to load prompt information" : `Failed to load version v${version}`);
    if (version !== null) clearVersion();
    else if (environment !== null) clearEnvironment();
  }, [isError, version, environment, clearVersion, clearEnvironment]);
  return isError && (version !== null || environment !== null);
}

function usePromptVersions(promptId: string, accessToken: string | null, environment: string | null) {
  const versionsQueryOptions: UseQueryOptions<PromptVersionsResponse> = {
    queryKey: ["promptVersions", promptId, environment],
    queryFn: () => getPromptVersions(accessToken ?? "", promptId, environment ?? undefined),
    enabled: !!accessToken && !!environment,
    retry: false,
  };
  const versionsQuery = useQuery(versionsQueryOptions);
  return { versionHistory: versionsQuery.data?.prompts ?? [], loadingVersions: versionsQuery.isLoading };
}

export function usePromptInfoData(promptId: string, accessToken: string | null, initialEnvironment?: string) {
  const urlState = usePromptInfoUrlState();
  const { environment, version } = urlState;
  const requestedEnv = environment ?? initialEnvironment;

  const infoQueryOptions: UseQueryOptions<PromptInfoResponse> = {
    queryKey: ["promptInfo", promptId, requestedEnv ?? null, version],
    queryFn: () => loadPromptInfo(accessToken ?? "", promptId, version, requestedEnv),
    enabled: !!accessToken,
    retry: false,
    placeholderData: (previous, previousQuery) => (previousQuery?.queryKey[1] === promptId ? previous : undefined),
  };
  const infoQuery = useQuery(infoQueryOptions);
  const isRecovering = useDropFailedSelection(infoQuery.isError, urlState);

  const response = infoQuery.data ?? null;
  const promptData = response?.prompt_spec ?? null;
  const environments = response?.environments ?? [];
  const selectedEnv = environments.length > 0 ? environment || promptData?.environment || environments[0] : null;

  const { versionHistory, loadingVersions } = usePromptVersions(promptId, accessToken, selectedEnv);

  return {
    rawApiResponse: response,
    promptTemplate: response?.raw_prompt_template ?? null,
    environments,
    loading: infoQuery.isFetching || isRecovering,
    selectedEnv,
    selectedVersion: version ?? (promptData?.version || null),
    versionHistory,
    loadingVersions,
    selectEnvironment: urlState.selectEnvironment,
    selectVersion: urlState.selectVersion,
  };
}
