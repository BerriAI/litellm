import { useQuery } from "@tanstack/react-query";
import { KeyResponse } from "../../components/key_team_helpers/key_list";
import { keyListCall } from "../../components/networking";

interface KeyListResponse {
  keys: KeyResponse[];
  total_count: number;
  current_page: number;
  total_pages: number;
}

interface UseKeysParams {
  accessToken: string;
  page?: number;
  pageSize?: number;
  enabled?: boolean;
}

interface UseKeysReturn {
  keys: KeyResponse[];
  pagination: {
    currentPage: number;
    totalPages: number;
    totalCount: number;
  };
  isLoading: boolean;
  error: Error | null;
  refetch: () => void;
  data: KeyListResponse | undefined;
}

export const useListKeys = ({
  accessToken,
  page = 1,
  pageSize = 100,
  enabled = true,
}: UseKeysParams): UseKeysReturn => {
  const query = useQuery<KeyListResponse, Error>({
    queryKey: ["keys", { page, pageSize }],
    queryFn: async () => {
      const data = await keyListCall(
        accessToken,
        null, // organizationID
        null, // teamID
        null, // selectedKeyAlias
        null, // userID
        null, // keyHash
        page,
        pageSize,
      );
      return data;
    },
    enabled: enabled && !!accessToken,
    staleTime: 30000, // Consider data fresh for 30 seconds
    refetchOnWindowFocus: false,
  });

  return {
    keys: query.data?.keys || [],
    pagination: {
      currentPage: query.data?.current_page || 1,
      totalPages: query.data?.total_pages || 0,
      totalCount: query.data?.total_count || 0,
    },
    isLoading: query.isLoading,
    error: query.error,
    refetch: query.refetch,
    data: query.data,
  };
};
