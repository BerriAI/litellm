import { useInfiniteQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { keyAliasesCall, type PaginatedKeyAliasResponse } from "@/components/networking";
import useAuthorized from "../useAuthorized";

const infiniteKeyAliasKeys = createQueryKeys("infiniteKeyAliases");

export const useInfiniteKeyAliases = (
  size: number = 50,
  search?: string,
) => {
  const { accessToken } = useAuthorized();
  return useInfiniteQuery<PaginatedKeyAliasResponse>({
    queryKey: infiniteKeyAliasKeys.list({
      filters: {
        size,
        ...(search && { search }),
      },
    }),
    queryFn: async ({ pageParam }) => {
      return await keyAliasesCall(
        accessToken!,
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
    enabled: Boolean(accessToken),
  });
};
