import { AutoRouterRoutingTestRequest } from "../networking";
import { ComplexityRouterConfigPayload } from "./build_complexity_router_config";
import { z } from "zod";

export const JEV_CONNECTION_TEST_PROMPT = "What is 2 plus 2?";

export const buildSavedJevConnectionTestRequest = (
  rawConfig: unknown,
  defaultModel?: string,
  routerName?: string,
  teamId?: string,
): AutoRouterRoutingTestRequest | undefined => {
  const parsed: unknown =
    typeof rawConfig === "string"
      ? (() => {
          try {
            return JSON.parse(rawConfig) as unknown;
          } catch {
            return undefined;
          }
        })()
      : rawConfig;
  const result = z
    .object({ classifier_type: z.literal("jev"), tiers: z.record(z.unknown()) })
    .passthrough()
    .safeParse(parsed);
  if (!result.success) return undefined;
  return {
    prompt: JEV_CONNECTION_TEST_PROMPT,
    complexity_router_config: result.data,
    ...(defaultModel && { default_model: defaultModel }),
    ...(routerName && { router_name: routerName }),
    ...(teamId && { team_id: teamId }),
  };
};

export interface BuildAutoRouterRoutingTestRequestParams {
  prompt: string;
  config: ComplexityRouterConfigPayload;
  defaultModel: string | undefined;
  routerName: string | undefined;
  teamId: string | undefined;
}

export const buildAutoRouterRoutingTestRequest = ({
  prompt,
  config,
  defaultModel,
  routerName,
  teamId,
}: BuildAutoRouterRoutingTestRequestParams): AutoRouterRoutingTestRequest => ({
  prompt,
  complexity_router_config: config,
  ...(defaultModel ? { default_model: defaultModel } : {}),
  ...(routerName?.trim() ? { router_name: routerName.trim() } : {}),
  ...(teamId ? { team_id: teamId } : {}),
});
