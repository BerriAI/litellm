import { modelCreateCall, Model } from "../networking";
import NotificationManager from "../molecules/notifications_manager";

export const handleAddAutoRouterSubmit = async (values: any, accessToken: string, form: any, callback?: () => void) => {
  try {
    let autoRouterConfig: any;

    if (values.model_type === "complexity_router") {
      autoRouterConfig = {
        model_name: values.auto_router_name,
        litellm_params: {
          model: `auto_router/complexity_router`,
          complexity_router_config: values.complexity_router_config,
          complexity_router_default_model: values.auto_router_default_model,
        },
        model_info: {},
      };
    } else {
      autoRouterConfig = {
        model_name: values.auto_router_name,
        litellm_params: {
          model: `auto_router/${values.auto_router_name}`,
          auto_router_config: JSON.stringify(values.auto_router_config),
          auto_router_default_model: values.auto_router_default_model,
        },
        model_info: {},
      };

      if (values.auto_router_embedding_model) {
        autoRouterConfig.litellm_params.auto_router_embedding_model = values.auto_router_embedding_model;
      }
    }

    if (values.team_id) {
      autoRouterConfig.model_info.team_id = values.team_id;
    }

    if (values.model_access_group && values.model_access_group.length > 0) {
      autoRouterConfig.model_info.access_groups = values.model_access_group;
    }

    await modelCreateCall(accessToken, autoRouterConfig as Model);

    const routerTypeName = values.model_type === "complexity_router" ? "Auto Router" : "Semantic Router";
    NotificationManager.success(`Successfully created ${routerTypeName}: ${values.auto_router_name}`);

    form.resetFields();

    if (callback) {
      callback();
    }
  } catch (error) {
    console.error("Failed to add auto router:", error);
    NotificationManager.fromBackend("Failed to add auto router: " + error);
  }
};
