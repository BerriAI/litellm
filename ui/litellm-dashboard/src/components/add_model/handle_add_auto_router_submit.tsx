import { modelCreateCall } from "../networking";
import { toast } from "@/lib/toast";
import type { ComplexityRouterConfigPayload } from "./build_complexity_router_config";
import type { CapabilityRouterConfigValue } from "./capability_router_config";

export interface AddAutoRouterValues {
  auto_router_name: string;
  model_type: "complexity_router" | "capability_router";
  auto_router_default_model?: string;
  complexity_router_config?: ComplexityRouterConfigPayload;
  capability_router_config?: CapabilityRouterConfigValue;
  team_id?: string;
  model_access_group?: string[];
}

export const handleAddAutoRouterSubmit = async (
  values: AddAutoRouterValues,
  accessToken: string,
  resetForm: () => void,
  callback?: () => void,
) => {
  try {
    const litellmParams =
      values.model_type === "capability_router"
        ? {
            model: "auto_router/capability_router",
            capability_router_config: values.capability_router_config!,
          }
        : {
            model: "auto_router/complexity_router",
            complexity_router_config: values.complexity_router_config!,
            complexity_router_default_model: values.auto_router_default_model,
          };
    const autoRouterConfig = {
      model_name: values.auto_router_name,
      litellm_params: litellmParams,
      model_info: {
        ...(values.team_id ? { team_id: values.team_id } : {}),
        ...(values.model_access_group?.length ? { access_groups: values.model_access_group } : {}),
      },
    };

    await modelCreateCall(accessToken, autoRouterConfig);

    toast.success(`Successfully created Auto Router: ${values.auto_router_name}`);

    resetForm();

    if (callback) {
      callback();
    }
  } catch (error) {
    console.error("Failed to add auto router:", error);
    toast.fromError("Failed to add auto router: " + error);
  }
};
