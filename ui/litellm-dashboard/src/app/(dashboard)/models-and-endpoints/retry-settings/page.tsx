"use client";

import { useCallback, useEffect, useState } from "react";
import ModelRetrySettingsTab from "@/app/(dashboard)/models-and-endpoints/components/ModelRetrySettingsTab";
import { getCallbacksCall } from "@/components/networking";
import { useUpdateRetryPolicy } from "@/app/(dashboard)/hooks/routerSettings/useUpdateRetryPolicy";
import { useModelDashboardData } from "@/app/(dashboard)/models-and-endpoints/useModelDashboardData";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import NotificationsManager from "@/components/molecules/notifications_manager";

interface RetryPolicyObject {
  [key: string]: { [retryPolicyKey: string]: number } | undefined;
}

interface GlobalRetryPolicyObject {
  [retryPolicyKey: string]: number;
}

interface RouterSettings {
  model_group_retry_policy?: RetryPolicyObject | null;
  retry_policy?: GlobalRetryPolicyObject | null;
  num_retries?: number | null;
}

export default function ModelRetrySettingsPage() {
  const { accessToken, userId: userID, userRole } = useAuthorized();
  const { availableModelGroups } = useModelDashboardData();
  const updateRetryPolicy = useUpdateRetryPolicy(accessToken);

  const [retryScope, setRetryScope] = useState<string | null>("global");
  const [modelGroupRetryPolicy, setModelGroupRetryPolicy] = useState<RetryPolicyObject | null>(null);
  const [globalRetryPolicy, setGlobalRetryPolicy] = useState<GlobalRetryPolicyObject | null>(null);
  const [defaultRetry, setDefaultRetry] = useState<number>(0);

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

  const applyRetrySettings = useCallback((routerSettings: RouterSettings) => {
    setModelGroupRetryPolicy(routerSettings.model_group_retry_policy ?? null);
    setGlobalRetryPolicy(routerSettings.retry_policy ?? null);
    setDefaultRetry(routerSettings.num_retries ?? 2);
  }, []);

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
    updateRetryPolicy.mutate(
      { retry_policy: globalRetryPolicy, model_group_retry_policy: modelGroupRetryPolicy },
      {
        onSuccess: () => {
          NotificationsManager.success("Retry settings saved successfully");
          void fetchRetrySettings().then((routerSettings) => {
            if (routerSettings) {
              applyRetrySettings(routerSettings);
            }
          });
        },
        onError: () => {
          NotificationsManager.fromBackend("Failed to save retry settings");
        },
      },
    );
  };

  return (
    <ModelRetrySettingsTab
      selectedModelGroup={retryScope}
      setSelectedModelGroup={setRetryScope}
      availableModelGroups={availableModelGroups}
      globalRetryPolicy={globalRetryPolicy}
      setGlobalRetryPolicy={setGlobalRetryPolicy}
      defaultRetry={defaultRetry}
      modelGroupRetryPolicy={modelGroupRetryPolicy}
      setModelGroupRetryPolicy={setModelGroupRetryPolicy}
      handleSaveRetrySettings={handleSaveRetrySettings}
      isSaving={updateRetryPolicy.isPending}
    />
  );
}
