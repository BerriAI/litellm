import { useQuery } from "@tanstack/react-query";
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

const modelKeys = createQueryKeys("models");
const modelHubKeys = createQueryKeys("modelHub");
const allProxyModelsKeys = createQueryKeys("allProxyModels");
const selectedTeamModelsKeys = createQueryKeys("selectedTeamModels");

export const useModelsInfo = () => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery({
    queryKey: modelKeys.list({
      filters: {
        ...(userId && { userId }),
        ...(userRole && { userRole }),
      },
    }),
    queryFn: async () => await modelInfoCall(accessToken!, userId!, userRole!),
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
    queryFn: async () => await modelAvailableCall(accessToken!, userId!, userRole!, true),
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
