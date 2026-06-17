import { modelCreateCall, Model } from "../networking";
import NotificationManager from "../molecules/notifications_manager";

interface AutoRouterFormValues {
  auto_router_name: string;
  auto_router_config?: unknown;
  auto_router_default_model?: string;
  auto_router_embedding_model?: string;
  complexity_router_config?: unknown;
  custom_embedding_model?: string;
  model_access_group?: string[];
  model_type?: string;
  team_id?: string;
}

export const buildAutoRouterModelConfig = (values: AutoRouterFormValues): Model => {
  if (values.model_type === "complexity_router") {
    return {
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
  }

  const litellmParams: Record<string, unknown> = {
    model: `auto_router/${values.auto_router_name}`,
    auto_router_config: JSON.stringify(values.auto_router_config),
    auto_router_default_model: values.auto_router_default_model,
  };

  if (values.auto_router_embedding_model && values.auto_router_embedding_model !== "custom") {
    litellmParams.auto_router_embedding_model = values.auto_router_embedding_model;
  } else if (values.custom_embedding_model) {
    litellmParams.auto_router_embedding_model = values.custom_embedding_model;
  }

  return {
    model_name: values.auto_router_name,
    litellm_params: litellmParams,
    model_info: {
      ...(values.team_id ? { team_id: values.team_id } : {}),
      ...(values.model_access_group?.length ? { access_groups: values.model_access_group } : {}),
    },
  };
};

export const handleAddAutoRouterSubmit = async (values: any, accessToken: string, form: any, callback?: () => void) => {
  try {
    console.log("=== AUTO ROUTER SUBMIT HANDLER CALLED ===");
    console.log("handling auto router submit for formValues:", values);
    console.log("Model type:", values.model_type);

    if (values.model_type === "complexity_router") {
      console.log("Creating complexity router configuration");
      console.log("Complexity router config:", values.complexity_router_config);
    } else {
      console.log("Creating semantic router configuration");
      console.log("Semantic router config (stringified):", JSON.stringify(values.auto_router_config));
    }

    const autoRouterConfig = buildAutoRouterModelConfig(values);

    console.log("Auto router configuration to be created:", autoRouterConfig);

    // Create the auto router using the same model creation endpoint
    console.log("Calling modelCreateCall...");
    const response: any = await modelCreateCall(accessToken, autoRouterConfig as Model);
    console.log(`response for auto router create call:`, response);

    // Show success notification
    const routerTypeName = values.model_type === "complexity_router" ? "Complexity Router" : "Semantic Router";
    NotificationManager.success(`Successfully created ${routerTypeName}: ${values.auto_router_name}`);

    // Reset the form
    form.resetFields();

    // Call the callback if provided (e.g., to close modal)
    if (callback) {
      callback();
    }
  } catch (error) {
    console.error("Failed to add auto router:", error);
    NotificationManager.fromBackend("Failed to add auto router: " + error);
  }
};
