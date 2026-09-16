import { useQuery, type UseQueryOptions } from "@tanstack/react-query";

import { getPromptInfo, getPromptVersions } from "@/components/networking";
import { toast } from "@/lib/toast";

import { usePromptInfoUrlState } from "./usePromptsUrlState";

type PromptInfoResponse = Awaited<ReturnType<typeof getPromptInfo>>;
type PromptVersionsResponse = Awaited<ReturnType<typeof getPromptVersions>>;

const loadPromptInfo = (accessToken: string, promptId: string, version: number | null, environment?: string) =>
  getPromptInfo(accessToken, version === null ? promptId : `${promptId}.v${version}`, environment).catch(
    (error: unknown) => {
      toast.fromError(version === null ? "Failed to load prompt information" : `Failed to load version v${version}`);
      console.error("Error fetching prompt info:", error);
      throw error;
    },
  );

export function usePromptInfoData(promptId: string, accessToken: string | null, initialEnvironment?: string) {
  const urlState = usePromptInfoUrlState();
  const requestedEnv = urlState.environment ?? initialEnvironment;

  const infoQueryOptions: UseQueryOptions<PromptInfoResponse> = {
    queryKey: ["promptInfo", promptId, requestedEnv ?? null, urlState.version],
    queryFn: () => loadPromptInfo(accessToken ?? "", promptId, urlState.version, requestedEnv),
    enabled: !!accessToken,
    retry: false,
    placeholderData: (previous, previousQuery) => (previousQuery?.queryKey[1] === promptId ? previous : undefined),
  };
  const infoQuery = useQuery(infoQueryOptions);
  const response = infoQuery.data ?? null;
  const promptData = response?.prompt_spec ?? null;
  const environments = response?.environments ?? [];
  const selectedEnv =
    environments.length > 0 ? urlState.environment || promptData?.environment || environments[0] : null;

  const versionsQueryOptions: UseQueryOptions<PromptVersionsResponse> = {
    queryKey: ["promptVersions", promptId, selectedEnv],
    queryFn: () => getPromptVersions(accessToken ?? "", promptId, selectedEnv ?? undefined),
    enabled: !!accessToken && !!selectedEnv,
    retry: false,
  };
  const versionsQuery = useQuery(versionsQueryOptions);

  return {
    rawApiResponse: response,
    promptData,
    promptTemplate: response?.raw_prompt_template ?? null,
    environments,
    loading: infoQuery.isFetching,
    selectedEnv,
    selectedVersion: urlState.version ?? (promptData?.version || null),
    versionHistory: versionsQuery.data?.prompts ?? [],
    loadingVersions: versionsQuery.isLoading,
    selectEnvironment: urlState.selectEnvironment,
    selectVersion: urlState.selectVersion,
  };
}
