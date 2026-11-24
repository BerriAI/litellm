import { useQuery } from "@tanstack/react-query";
import { useControlPlane } from "@/contexts/ControlPlaneContext";
import { createQueryKeys } from "../common/queryKeysFactory";
import { fetchMCPToolsets } from "@/components/networking";
import { MCPToolset } from "@/components/mcp_tools/types";
import useAuthorized from "../useAuthorized";

const mcpToolsetKeys = createQueryKeys("mcpToolsets");

export const useMCPToolsets = () => {
  const controlPlane = useControlPlane();
  const { accessToken } = useAuthorized();
  return useQuery<MCPToolset[]>({
    queryKey: mcpToolsetKeys.list(),
    queryFn: async () => await fetchMCPToolsets(accessToken!),
    enabled: !controlPlane && !!accessToken,
  });
};
