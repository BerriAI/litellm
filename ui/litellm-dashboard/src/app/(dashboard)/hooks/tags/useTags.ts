import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { fetchClient, type UntypedApiResponse } from "@/lib/http/api";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";

const tagKeys = createQueryKeys("tags");

export const useTags = (): UseQueryResult<UntypedApiResponse> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<UntypedApiResponse>({
    queryKey: tagKeys.list({}),
    queryFn: async () => {
      const { data } = await fetchClient.GET("/tag/list");
      return (data ?? {}) as UntypedApiResponse;
    },
    enabled: Boolean(accessToken && userId && userRole),
  });
};
