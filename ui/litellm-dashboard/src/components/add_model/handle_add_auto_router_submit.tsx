import { modelCreateCall, Model } from "../networking";
import NotificationManager from "../molecules/notifications_manager";

export const handleAddAutoRouterSubmit = async (values: any, accessToken: string, form: any, callback?: () => void) => {
  try {
    console.log("=== AUTO ROUTER SUBMIT HANDLER CALLED ===");
    console.log("handling auto router submit for formValues:", values);
    console.log("Access token:", accessToken ? "Present" : "Missing");
    console.log("Form:", form ? "Present" : "Missing");
    console.log("Callback:", callback ? "Present" : "Missing");

    // Create auto router configuration
    const autoRouterConfig: any = {
      model_name: values.auto_router_name,
      litellm_params: {
        model: `auto_router/${values.auto_router_name}`,
        auto_router_config: JSON.stringify(values.auto_router_config), // Convert JSON object to string as expected by backend
        auto_router_default_model: values.auto_router_default_model,
      },
      model_info: {},
    };

    // Add optional embedding model if provided
    if (values.auto_router_embedding_model && values.auto_router_embedding_model !== "custom") {
      autoRouterConfig.litellm_params.auto_router_embedding_model = values.auto_router_embedding_model;
    } else if (values.custom_embedding_model) {
      autoRouterConfig.litellm_params.auto_router_embedding_model = values.custom_embedding_model;
    }

    // Add team information if provided
    if (values.team_id) {
      autoRouterConfig.model_info.team_id = values.team_id;
    }

    // Add model access groups if provided
    if (values.model_access_group && values.model_access_group.length > 0) {
      autoRouterConfig.model_info.access_groups = values.model_access_group;
    }

    console.log("Auto router configuration to be created:", autoRouterConfig);
    console.log("Auto router config (stringified):", autoRouterConfig.litellm_params.auto_router_config);

    // Create the auto router using the same model creation endpoint
    console.log("Calling modelCreateCall with:", {
      accessToken: accessToken ? "Present" : "Missing",
      config: autoRouterConfig,
    });
    const response: any = await modelCreateCall(accessToken, autoRouterConfig as Model);
    console.log(`response for auto router create call:`, response);

    // Reset the form
    form.resetFields();
  } catch (error) {
    console.error("Failed to add auto router:", error);
    NotificationManager.fromBackend("Failed to add auto router: " + error);
  }
};
