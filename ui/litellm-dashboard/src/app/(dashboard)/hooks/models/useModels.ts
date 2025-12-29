import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { modelInfoCall, modelHubCall } from "@/components/networking";
import useAuthorized from "../useAuthorized";
const modelKeys = createQueryKeys("models");
const modelHubKeys = createQueryKeys("modelHub");

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
