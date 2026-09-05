import { modelCreateCall } from "../networking";
import { toast } from "@/lib/toast";
import type { ComplexityRouterConfigPayload } from "./build_complexity_router_config";

export interface AddAutoRouterValues {
  auto_router_name: string;
  auto_router_default_model: string | undefined;
  model_type: "complexity_router";
  complexity_router_config: ComplexityRouterConfigPayload;
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
    const autoRouterConfig = {
      model_name: values.auto_router_name,
      litellm_params: {
        model: "auto_router/complexity_router",
        complexity_router_config: values.complexity_router_config,
        complexity_router_default_model: values.auto_router_default_model,
      },
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
