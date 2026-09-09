import { modelCreateCall, Model } from "../networking";
import { toast } from "@/lib/toast";

export interface AddAdeptRouterValues {
  adept_router_name: string;
  adept_router_default_model: string;
  adept_router_tag_prefix?: string;
  adept_router_conversations_threshold?: number | null;
  adept_router_trainer_url?: string;
  adept_router_pg_host?: string;
  adept_router_pg_port?: number | null;
  adept_router_pg_database?: string;
  adept_router_pg_user?: string;
  adept_router_pg_password?: string;
  adept_router_pg_ssl_mode?: string;
  team_id?: string;
  model_access_group?: string[];
}

// Drops undefined / empty-string entries so `litellm_params` on the backend only carries the
// keys the operator actually set; keeps the create payload aligned with the Pydantic defaults.
const dropEmpty = <T extends Record<string, unknown>>(values: T): Partial<T> =>
  Object.fromEntries(
    Object.entries(values).filter(([, value]) => value !== undefined && value !== "" && value !== null),
  ) as Partial<T>;

export const handleAddAdeptRouterSubmit = async (
  values: AddAdeptRouterValues,
  accessToken: string,
  resetForm: () => void,
  callback?: () => void,
) => {
  try {
    const rawLitellmParams = {
      model: `adept/${values.adept_router_name}`,
      adept_router_default_model: values.adept_router_default_model,
      adept_router_tag_prefix: values.adept_router_tag_prefix,
      adept_router_conversations_threshold: values.adept_router_conversations_threshold ?? undefined,
      adept_router_trainer_url: values.adept_router_trainer_url,
      adept_router_pg_host: values.adept_router_pg_host,
      adept_router_pg_port: values.adept_router_pg_port ?? undefined,
      adept_router_pg_database: values.adept_router_pg_database,
      adept_router_pg_user: values.adept_router_pg_user,
      adept_router_pg_password: values.adept_router_pg_password,
      adept_router_pg_ssl_mode: values.adept_router_pg_ssl_mode,
    };
    const litellmParams = dropEmpty(rawLitellmParams);

    const adeptConfig = {
      model_name: values.adept_router_name,
      litellm_params: litellmParams,
      model_info: {
        ...(values.team_id ? { team_id: values.team_id } : {}),
        ...(values.model_access_group?.length ? { access_groups: values.model_access_group } : {}),
      },
    };

    await modelCreateCall(accessToken, adeptConfig as unknown as Model);
    toast.success(`Successfully created ADEPT Router: ${values.adept_router_name}`);
    resetForm();
    callback?.();
  } catch (error) {
    console.error("Failed to add ADEPT router:", error);
    toast.fromError("Failed to add ADEPT router: " + error);
  }
};
