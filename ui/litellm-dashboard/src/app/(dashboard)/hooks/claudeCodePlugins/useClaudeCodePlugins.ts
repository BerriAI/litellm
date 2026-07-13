import { useQuery } from "@tanstack/react-query";
import { createQueryKeys } from "../common/queryKeysFactory";
import { getClaudeCodePluginsList } from "@/components/networking";
import { ListPluginsResponse, PluginListItem } from "@/components/claude_code_plugins/types";
import useAuthorized from "../useAuthorized";

const claudeCodePluginKeys = createQueryKeys("claudeCodePlugins");

export const useClaudeCodePlugins = () => {
  const { accessToken } = useAuthorized();
  return useQuery<PluginListItem[]>({
    queryKey: claudeCodePluginKeys.list(),
    queryFn: async () => {
      const response: ListPluginsResponse = await getClaudeCodePluginsList(accessToken!, false);
      return response.plugins;
    },
    enabled: !!accessToken,
  });
};

export { claudeCodePluginKeys };
