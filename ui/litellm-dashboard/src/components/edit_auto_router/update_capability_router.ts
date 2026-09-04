import { modelPatchUpdateCall, validateAutoRouterConfig } from "../networking";
import { dryRunRejection } from "../add_model/build_complexity_router_config";
import { capabilityRouterConfigError, type CapabilityRouterConfigValue } from "../add_model/capability_router_config";

interface CapabilityRouterModelInfo extends Record<string, unknown> {
  id: string;
  team_id?: string;
}

interface CapabilityRouterModelData extends Record<string, unknown> {
  litellm_params: Record<string, unknown>;
  model_info: CapabilityRouterModelInfo;
}

interface CapabilityRouterEditValues {
  auto_router_name: string;
  model_access_group?: string[];
}

type CapabilityRouterUpdateResult =
  | { kind: "error"; message: string }
  | { kind: "success"; updatedModel: CapabilityRouterModelData };

interface UpdateCapabilityRouterParams {
  accessToken: string;
  config: CapabilityRouterConfigValue;
  modelData: CapabilityRouterModelData;
  values: CapabilityRouterEditValues;
}

export const updateCapabilityRouter = async ({
  accessToken,
  config,
  modelData,
  values,
}: UpdateCapabilityRouterParams): Promise<CapabilityRouterUpdateResult> => {
  const validationError = capabilityRouterConfigError(config);
  if (validationError) return { kind: "error", message: validationError };

  const serverVerdict = await validateAutoRouterConfig(
    accessToken,
    config as unknown as Record<string, unknown>,
    modelData.model_info.team_id,
    "capability",
  );
  const serverError = dryRunRejection(serverVerdict);
  if (serverError) return { kind: "error", message: serverError };

  const updatedLitellmParams = { ...modelData.litellm_params, capability_router_config: config };
  const updatedModelInfo = { ...modelData.model_info, access_groups: values.model_access_group ?? [] };
  const updatedModel = {
    ...modelData,
    model_name: values.auto_router_name,
    litellm_params: updatedLitellmParams,
    model_info: updatedModelInfo,
  };
  await modelPatchUpdateCall(accessToken, updatedModel, modelData.model_info.id);
  return { kind: "success", updatedModel };
};
