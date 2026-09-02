import { AutoRouterRoutingTestRequest } from "../networking";
import { ComplexityRouterConfigPayload } from "./build_complexity_router_config";
import { CapabilityRouterConfigValue } from "./capability_router_config";

export interface BuildAutoRouterRoutingTestRequestParams {
  prompt: string;
  config?: ComplexityRouterConfigPayload;
  capabilityConfig?: CapabilityRouterConfigValue;
  defaultModel: string | undefined;
  routerName: string | undefined;
  teamId: string | undefined;
}

export const buildAutoRouterRoutingTestRequest = ({
  prompt,
  config,
  capabilityConfig,
  defaultModel,
  routerName,
  teamId,
}: BuildAutoRouterRoutingTestRequestParams): AutoRouterRoutingTestRequest => ({
  prompt,
  ...(capabilityConfig
    ? { capability_router_config: capabilityConfig as unknown as Record<string, unknown> }
    : { complexity_router_config: config }),
  ...(defaultModel ? { default_model: defaultModel } : {}),
  ...(routerName?.trim() ? { router_name: routerName.trim() } : {}),
  ...(teamId ? { team_id: teamId } : {}),
});
