import { useQuery, useInfiniteQuery } from "@tanstack/react-query";
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
const allProxyModelsKeys = createQueryKeys("allProxyModels");
const selectedTeamModelsKeys = createQueryKeys("selectedTeamModels");
const infiniteModelKeys = createQueryKeys("infiniteModels");

export const useModelsInfo = (page: number = 1, size: number = 50, search?: string, modelId?: string, teamId?: string, sortBy?: string, sortOrder?: string) => {
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
    queryFn: async () => await modelInfoCall(accessToken!, userId!, userRole!, page, size, search, modelId, teamId, sortBy, sortOrder),
    enabled: Boolean(accessToken && userId && userRole),
  });
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

export const useSelectedTeamModels = (teamID: string | null) => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<AllProxyModelsResponse>({
    queryKey: selectedTeamModelsKeys.list({}),
    queryFn: async () => await modelAvailableCall(accessToken!, userId!, userRole!, true, teamID!),
    enabled: Boolean(accessToken && userId && userRole && teamID),
  });
};

export const useInfiniteModelInfo = (
  size: number = 50,
  search?: string,
) => {
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
      return await modelInfoCall(
        accessToken!,
        userId!,
        userRole!,
        pageParam as number,
        size,
        search,
      );
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
