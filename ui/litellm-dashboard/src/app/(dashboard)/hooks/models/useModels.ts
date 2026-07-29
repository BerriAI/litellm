import { useQuery, useInfiniteQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { modelInfoCall, modelHubCall, modelAvailableCall } from "@/components/networking";
import useAuthorized from "../useAuthorized";

export interface ProxyModel {
  id: string;
  object: string;
  created: number;
  owned_by: string;
}

export interface AllProxyModelsResponse {
  data: ProxyModel[];
}

export interface PaginatedModelInfoResponse {
  data: any[];
  total_count: number;
  current_page: number;
  total_pages: number;
  size: number;
}

const modelKeys = createQueryKeys("models");
const modelHubKeys = createQueryKeys("modelHub");
const autoRouterKeys = createQueryKeys("autoRouterModelGroups");
const allProxyModelsKeys = createQueryKeys("allProxyModels");
const selectedTeamModelsKeys = createQueryKeys("selectedTeamModels");
const infiniteModelKeys = createQueryKeys("infiniteModels");
const userModelsKeys = createQueryKeys("userModels");

export const useModelsInfo = (
  page: number = 1,
  size: number = 50,
  search?: string,
  modelId?: string,
  teamId?: string,
  sortBy?: string,
  sortOrder?: string,
) => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<PaginatedModelInfoResponse>({
    queryKey: modelKeys.list({
      filters: {
        ...(userId && { userId }),
        ...(userRole && { userRole }),
        page,
        size,
        ...(search && { search }),
        ...(modelId && { modelId }),
        ...(teamId && { teamId }),
        ...(sortBy && { sortBy }),
        ...(sortOrder && { sortOrder }),
      },
    }),
    queryFn: async () =>
      await modelInfoCall(accessToken!, userId!, userRole!, page, size, search, modelId, teamId, sortBy, sortOrder),
    enabled: Boolean(accessToken && userId && userRole),
  });
};

const AUTO_ROUTER_MODEL_PREFIX = "auto_router/";
const AUTO_ROUTER_LOOKUP_PAGE_SIZE = 1000;
const NO_AUTO_ROUTERS: ReadonlySet<string> = new Set<string>();

export interface AutoRouterCandidateDeployment {
  model_name?: string | null;
  litellm_params?: { model?: string | null } | null;
}

export const isAutoRouterDeployment = (deployment: AutoRouterCandidateDeployment): boolean =>
  Boolean(deployment?.litellm_params?.model?.startsWith(AUTO_ROUTER_MODEL_PREFIX));

export const selectAutoRouterModelGroups = (deployments: AutoRouterCandidateDeployment[]): ReadonlySet<string> =>
  new Set(
    deployments
      .filter(isAutoRouterDeployment)
      .map((deployment) => deployment.model_name)
      .filter((modelName): modelName is string => Boolean(modelName)),
  );

const fetchAllModelDeployments = async (
  accessToken: string,
  userId: string,
  userRole: string,
): Promise<AutoRouterCandidateDeployment[]> => {
  const firstPage: PaginatedModelInfoResponse = await modelInfoCall(
    accessToken,
    userId,
    userRole,
    1,
    AUTO_ROUTER_LOOKUP_PAGE_SIZE,
  );
  const totalPages = firstPage?.total_pages ?? 1;
  const remainingPages = await Promise.all(
    Array.from({ length: Math.max(0, totalPages - 1) }, (_unused, index) =>
      modelInfoCall(accessToken, userId, userRole, index + 2, AUTO_ROUTER_LOOKUP_PAGE_SIZE),
    ),
  );
  return [firstPage, ...remainingPages].flatMap(
    (page: PaginatedModelInfoResponse) => page?.data ?? [],
  ) as AutoRouterCandidateDeployment[];
};

export const useAutoRouterModelGroups = (): ReadonlySet<string> => {
  const { accessToken, userId, userRole } = useAuthorized();
  const { data } = useQuery<AutoRouterCandidateDeployment[], Error, ReadonlySet<string>>({
    queryKey: autoRouterKeys.list({
      filters: {
        ...(userId && { userId }),
        ...(userRole && { userRole }),
      },
    }),
    queryFn: async () => await fetchAllModelDeployments(accessToken!, userId!, userRole!),
    enabled: Boolean(accessToken && userId && userRole),
    select: selectAutoRouterModelGroups,
  });
  return data ?? NO_AUTO_ROUTERS;
};

export const useModelHub = () => {
  const { accessToken } = useAuthorized();
  return useQuery({
    queryKey: modelHubKeys.list({}),
    queryFn: async () => await modelHubCall(accessToken!),
    enabled: Boolean(accessToken),
  });
};

export const useAllProxyModels = () => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<AllProxyModelsResponse>({
    queryKey: allProxyModelsKeys.list({}),
    queryFn: async () => await modelAvailableCall(accessToken!, userId!, userRole!, true, null, true, false, "expand"),
    enabled: Boolean(accessToken && userId && userRole),
  });
};

export const useUserModels = (): UseQueryResult<string[]> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<string[]>({
    queryKey: userModelsKeys.list({}),
    queryFn: async () => {
      const response = await modelAvailableCall(accessToken!, userId!, userRole!);
      return response["data"].map((model: { id: string }) => model.id);
    },
    enabled: Boolean(accessToken && userId && userRole),
  });
};

export const useSelectedTeamModels = (teamID: string | null) => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<AllProxyModelsResponse>({
    queryKey: selectedTeamModelsKeys.list({}),
    queryFn: async () => await modelAvailableCall(accessToken!, userId!, userRole!, true, teamID!),
    enabled: Boolean(accessToken && userId && userRole && teamID),
  });
};

export const useInfiniteModelInfo = (size: number = 50, search?: string) => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useInfiniteQuery<PaginatedModelInfoResponse>({
    queryKey: infiniteModelKeys.list({
      filters: {
        ...(userId && { userId }),
        ...(userRole && { userRole }),
        size,
        ...(search && { search }),
      },
    }),
    queryFn: async ({ pageParam }) => {
      return await modelInfoCall(accessToken!, userId!, userRole!, pageParam as number, size, search);
    },
    initialPageParam: 1,
    getNextPageParam: (lastPage) => {
      if (lastPage.current_page < lastPage.total_pages) {
        return lastPage.current_page + 1;
      }
      return undefined;
    },
    enabled: Boolean(accessToken && userId && userRole),
  });
};
