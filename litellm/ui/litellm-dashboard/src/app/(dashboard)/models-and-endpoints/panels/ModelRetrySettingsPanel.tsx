"use client";

import { useCallback, useEffect, useState, type Dispatch, type SetStateAction } from "react";
import { useController } from "react-hook-form";
import { useTranslation } from "react-i18next";
import ModelRetrySettingsTab from "@/app/(dashboard)/models-and-endpoints/components/ModelRetrySettingsTab";
import { getCallbacksCall } from "@/components/networking";
import { useUpdateRetryPolicy } from "@/app/(dashboard)/hooks/routerSettings/useUpdateRetryPolicy";
import { useModelDashboardData } from "@/app/(dashboard)/models-and-endpoints/useModelDashboardData";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { toast } from "@/lib/toast";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toRetryPolicyPayload, toRetrySettings, type RouterRetrySettings } from "./model-retry-settings/mapper";
import { RetryPolicyDraft, retrySettingsSchema } from "./model-retry-settings/schema";

const initialDraft = toRetrySettings({}).draft;

export default function ModelRetrySettingsPanel() {
  const { t } = useTranslation();
  const { accessToken, userId: userID, userRole } = useAuthorized();
  const { availableModelGroups } = useModelDashboardData();
  const updateRetryPolicy = useUpdateRetryPolicy(accessToken);

  const { control, getValues, setValue } = useZodForm(retrySettingsSchema, {
    defaultValues: { scope: "global", draft: initialDraft },
  });
  const { field: scope } = useController({ control, name: "scope" });
  const {
    field: {
      value: { policies },
    },
  } = useController({ control, name: "draft" });
  const [defaultRetry, setDefaultRetry] = useState<number>(0);

  const setPolicy = <Key extends keyof typeof policies>(key: Key, update: SetStateAction<(typeof policies)[Key]>) => {
    const current = getValues("draft").policies;
    setValue(
      "draft",
      new RetryPolicyDraft({
        ...current,
        [key]: typeof update === "function" ? update(current[key]) : update,
      }),
    );
  };
  const setGlobalRetryPolicy: Dispatch<SetStateAction<typeof policies.retry_policy>> = (update) =>
    setPolicy("retry_policy", update);
  const setModelGroupRetryPolicy: Dispatch<SetStateAction<typeof policies.model_group_retry_policy>> = (update) =>
    setPolicy("model_group_retry_policy", update);

  const fetchRetrySettings = useCallback(async () => {
    if (!accessToken || !userID || !userRole) {
      return null;
    }
    try {
      const info = await getCallbacksCall(accessToken, userID, userRole);
      return info.router_settings;
    } catch (error) {
      console.error("Error fetching router settings:", error);
      return null;
    }
  }, [accessToken, userID, userRole]);

  const applyRetrySettings = useCallback(
    (routerSettings: RouterRetrySettings) => {
      const { draft, defaultRetry } = toRetrySettings(routerSettings);
      setValue("draft", draft);
      setDefaultRetry(defaultRetry);
    },
    [setValue],
  );

  useEffect(() => {
    let active = true;
    void (async () => {
      const routerSettings = await fetchRetrySettings();
      if (active && routerSettings) {
        applyRetrySettings(routerSettings);
      }
    })();
    return () => {
      active = false;
    };
  }, [fetchRetrySettings, applyRetrySettings]);

  const handleSaveRetrySettings = () => {
    const values = retrySettingsSchema.parse(getValues());
    updateRetryPolicy.mutate(toRetryPolicyPayload(values.draft), {
      onSuccess: () => {
        toast.success(t("modelsAndEndpoints.modelRetrySettingsPanel.toasts.saved"));
        void fetchRetrySettings().then((routerSettings) => {
          if (routerSettings) {
            applyRetrySettings(routerSettings);
          }
        });
      },
      onError: () => {
        toast.fromError(t("modelsAndEndpoints.modelRetrySettingsPanel.toasts.saveFailed"));
      },
    });
  };

  return (
    <ModelRetrySettingsTab
      selectedModelGroup={scope.value}
      setSelectedModelGroup={scope.onChange}
      availableModelGroups={availableModelGroups}
      globalRetryPolicy={policies.retry_policy}
      setGlobalRetryPolicy={setGlobalRetryPolicy}
      defaultRetry={defaultRetry}
      modelGroupRetryPolicy={policies.model_group_retry_policy}
      setModelGroupRetryPolicy={setModelGroupRetryPolicy}
      handleSaveRetrySettings={handleSaveRetrySettings}
      isSaving={updateRetryPolicy.isPending}
    />
  );
}
