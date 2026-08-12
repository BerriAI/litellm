import type { FormInstance } from "antd";
import { modelCreateCall, Model } from "../networking";
import NotificationManager from "../molecules/notifications_manager";

interface AdeptRouterFormValues {
  adept_router_name: string;
  adept_router_default_model: string;
  adept_router_tag_prefix?: string;
  adept_router_conversations_threshold?: number;
  adept_router_trainer_url?: string;
  adept_router_pg_host?: string;
  adept_router_pg_port?: number;
  adept_router_pg_database?: string;
  adept_router_pg_user?: string;
  adept_router_pg_password?: string;
  team_id?: string;
  model_access_group?: string[];
}

export const handleAddAdeptRouterSubmit = async (
  values: AdeptRouterFormValues,
  accessToken: string,
  form: FormInstance,
  callback?: () => void,
) => {
  try {
    const modelInfo: Record<string, unknown> = {};

    if (values.team_id) {
      modelInfo.team_id = values.team_id;
    }

    if (values.model_access_group && values.model_access_group.length > 0) {
      modelInfo.access_groups = values.model_access_group;
    }

    const adeptConfig = {
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
      model_info: modelInfo,
    };

    await modelCreateCall(accessToken, adeptConfig as unknown as Model);
    form.resetFields();
    callback?.();
  } catch (error) {
    NotificationManager.fromBackend("Failed to add ADEPT router: " + error);
  }
};
