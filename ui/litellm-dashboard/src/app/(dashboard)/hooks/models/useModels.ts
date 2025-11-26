import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { modelInfoCall, modelHubCall } from "@/components/networking";

const modelKeys = createQueryKeys("models");
const modelHubKeys = createQueryKeys("modelHub");

export const useModelsInfo = (accessToken: string | null, userID: string | null, userRole: string | null) => {
  return useQuery({
    queryKey: modelKeys.list({
      filters: {
        ...(userID && { userID }),
        ...(userRole && { userRole }),
      },
    }),
    queryFn: async () => await modelInfoCall(accessToken!, userID!, userRole!),
    enabled: Boolean(accessToken && userID && userRole),
  });
};

export const useModelHub = (accessToken: string | null) => {
  return useQuery({
    queryKey: modelHubKeys.list({}),
    queryFn: async () => await modelHubCall(accessToken!),
    enabled: Boolean(accessToken),
  });
};
