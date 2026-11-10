import { modelCreateCall } from "../networking";
import { toast } from "@/lib/toast";
import i18n from "@/i18n/i18n";
import type { ComplexityRouterConfigPayload } from "./build_complexity_router_config";
import type { AutoRouterCompressionLitellmParams } from "./buildAutoRouterCompression";
import { switchyardRouterPayload, switchyardSaveError, type SwitchyardConfig } from "./switchyard_config";

export interface AddAutoRouterValues extends AutoRouterCompressionLitellmParams {
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
        ...(values.auto_router_routing_compression === undefined
          ? {}
          : {
              auto_router_routing_compression: values.auto_router_routing_compression,
            }),
        ...(values.auto_router_model_compression === undefined
          ? {}
          : {
              auto_router_model_compression: values.auto_router_model_compression,
            }),
      },
      model_info: {
        ...(values.team_id ? { team_id: values.team_id } : {}),
        ...(values.model_access_group?.length ? { access_groups: values.model_access_group } : {}),
      },
    };

    await modelCreateCall(accessToken, autoRouterConfig);

    toast.success(i18n.t("addModel.handleAddAutoRouterSubmit.created", { name: values.auto_router_name }));

    resetForm();

    if (callback) {
      callback();
    }
  } catch (error) {
    console.error("Failed to add auto router:", error);
    toast.fromError(i18n.t("addModel.handleAddAutoRouterSubmit.failed", { error: String(error) }));
  }
};

export interface SwitchyardModelInfo {
  team_id?: string;
  access_groups?: string[];
}

export const handleAddSwitchyardSubmit = async (
  draft: { name: string; config: SwitchyardConfig; toolSemanticsError: string | null; modelInfo: SwitchyardModelInfo },
  accessToken: string,
  resetForm: () => void,
  callback: () => void,
) => {
  const invalid = draft.toolSemanticsError ?? switchyardSaveError(draft.name, draft.config);
  if (invalid) return toast.fromError(invalid);
  try {
    await modelCreateCall(accessToken, {
      ...switchyardRouterPayload(draft.name, draft.config),
      model_info: draft.modelInfo,
    });
    toast.success(i18n.t("addModel.handleAddAutoRouterSubmit.created", { name: draft.name }));
    resetForm();
    callback();
  } catch (error) {
    toast.fromError(i18n.t("addModel.handleAddAutoRouterSubmit.failed", { error: String(error) }));
  }
};
