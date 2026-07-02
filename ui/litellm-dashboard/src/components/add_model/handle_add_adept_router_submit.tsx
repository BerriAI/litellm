import { modelCreateCall, Model } from "../networking";
import NotificationManager from "../molecules/notifications_manager";

export const handleAddAdeptRouterSubmit = async (
  values: any,
  accessToken: string,
  form: any,
  callback?: () => void,
) => {
  try {
    const adeptConfig: any = {
      model_name: values.adept_router_name,
      litellm_params: {
        model: `adept/${values.adept_router_name}`,
        adept_router_default_model: values.adept_router_default_model,
        adept_router_tag_prefix: values.adept_router_tag_prefix || undefined,
        adept_router_conversations_threshold: values.adept_router_conversations_threshold || undefined,
        adept_router_trainer_url: values.adept_router_trainer_url || undefined,
        adept_router_pg_host: values.adept_router_pg_host || undefined,
        adept_router_pg_port: values.adept_router_pg_port || undefined,
        adept_router_pg_database: values.adept_router_pg_database || undefined,
        adept_router_pg_user: values.adept_router_pg_user || undefined,
        adept_router_pg_password: values.adept_router_pg_password || undefined,
      },
      model_info: {},
    };

    if (values.team_id) {
      adeptConfig.model_info.team_id = values.team_id;
    }

    if (values.model_access_group && values.model_access_group.length > 0) {
      adeptConfig.model_info.access_groups = values.model_access_group;
    }

    await modelCreateCall(accessToken, adeptConfig as Model);
    form.resetFields();
    callback?.();
  } catch (error) {
    NotificationManager.fromBackend("Failed to add ADEPT router: " + error);
  }
};
