"use client";

import { parseAsString, useQueryState } from "nuqs";
import { useCallback, useEffect, useMemo, useState } from "react";
import ModelRetrySettingsTab from "@/app/(dashboard)/models-and-endpoints/components/ModelRetrySettingsTab";
import { getCallbacksCall } from "@/components/networking";
import { useUpdateRetryPolicy } from "@/app/(dashboard)/hooks/routerSettings/useUpdateRetryPolicy";
import { useModelDashboardData } from "@/app/(dashboard)/models-and-endpoints/useModelDashboardData";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { toast } from "@/lib/toast";
import { useUrlTab } from "@/hooks/useUrlTab";

const GLOBAL_SCOPE = "global";
const RETRY_SCOPE_KEY = "retry_scope";

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

export default function ModelRetrySettingsPanel() {
  const { accessToken, userId: userID, userRole } = useAuthorized();
  const { availableModelGroups } = useModelDashboardData();
  const updateRetryPolicy = useUpdateRetryPolicy(accessToken);

  const [requestedScope] = useQueryState(RETRY_SCOPE_KEY, parseAsString.withDefault(GLOBAL_SCOPE));
  const allowedScopes = useMemo(
    () => [GLOBAL_SCOPE, ...(availableModelGroups.length > 0 ? availableModelGroups : [requestedScope])],
    [requestedScope, availableModelGroups],
  );
  const [retryScope, setRetryScope] = useUrlTab(allowedScopes, GLOBAL_SCOPE, RETRY_SCOPE_KEY);
  const selectRetryScope = useCallback((scope: string | null) => setRetryScope(scope ?? GLOBAL_SCOPE), [setRetryScope]);
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
          toast.success("Retry settings saved successfully");
          void fetchRetrySettings().then((routerSettings) => {
            if (routerSettings) {
              applyRetrySettings(routerSettings);
            }
          });
        },
        onError: () => {
          toast.fromError("Failed to save retry settings");
        },
      },
    );
  };

  return (
    <ModelRetrySettingsTab
      selectedModelGroup={retryScope}
      setSelectedModelGroup={selectRetryScope}
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
