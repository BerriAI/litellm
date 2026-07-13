import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { getClaudeCodeMarketplaces } from "@/components/networking";
import { ListMarketplacesResponse, MarketplaceSource } from "@/components/claude_code_plugins/types";
import useAuthorized from "../useAuthorized";

const claudeCodeMarketplaceKeys = createQueryKeys("claudeCodeMarketplaces");

export const useClaudeCodeMarketplaces = () => {
  const { accessToken } = useAuthorized();
  return useQuery<MarketplaceSource[]>({
    queryKey: claudeCodeMarketplaceKeys.list(),
    queryFn: async () => {
      const response: ListMarketplacesResponse = await getClaudeCodeMarketplaces(accessToken!);
      return response.marketplaces;
    },
    enabled: !!accessToken,
  });
};

export { claudeCodeMarketplaceKeys };
