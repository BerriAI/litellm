import { useQuery, UseQueryOptions } from "@tanstack/react-query";
import { createQueryKeys } from "@/app/(dashboard)/hooks/common/queryKeysFactory";
import { getProxyBaseUrl } from "@/components/networking";
import { ApiError, createApiClient } from "@/lib/http/client";

const anonymousApiClient = createApiClient({ getBaseUrl: getProxyBaseUrl });

export const fetchClaudeCodeMarketplaceRequiresKey = async (): Promise<boolean> => {
  try {
    await anonymousApiClient.get("/claude-code/marketplace.json", { credentials: "omit" });
    return false;
  } catch (error) {
    if (error instanceof ApiError && error.status === 401) {
      return true;
    }
    throw error;
  }
};

export const claudeCodeMarketplaceKeys = createQueryKeys("claudeCodeMarketplace");

const requiresKeyQueryOptions: UseQueryOptions<boolean> = {
  queryKey: claudeCodeMarketplaceKeys.detail("requiresKey"),
  queryFn: fetchClaudeCodeMarketplaceRequiresKey,
  staleTime: 60 * 1000,
  retry: 1,
};

export const useClaudeCodeMarketplaceRequiresKey = () => useQuery<boolean>(requiresKeyQueryOptions);
