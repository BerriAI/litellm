import { useQuery, UseQueryResult } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { tagListCall } from "@/components/networking";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { TagListResponse } from "@/components/tag_management/types";

const tagKeys = createQueryKeys("tags");

export const useTags = (): UseQueryResult<TagListResponse> => {
  const { accessToken, userId, userRole } = useAuthorized();
  return useQuery<TagListResponse>({
    queryKey: tagKeys.list({}),
    queryFn: async () => await tagListCall(accessToken!),
    enabled: Boolean(accessToken && userId && userRole),
  });
};
