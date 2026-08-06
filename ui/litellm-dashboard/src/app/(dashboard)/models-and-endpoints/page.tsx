"use client";

import { useMemo, useState } from "react";
import { Tabs } from "antd";
import { RefreshIcon } from "@heroicons/react/outline";
import { useQueryClient } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { all_admin_roles, internalUserRoles } from "@/utils/roles";
import { canCreateModels } from "@/utils/modelPermissions";
import BetaBadge from "@/components/BetaBadge";
import CostOptimizationFeedbackBanner from "@/components/molecules/cost_optimization_feedback_banner";
import ModelInfoView from "@/components/model_info_view";
import TeamInfoView from "@/components/team/TeamInfo";
import { useModelDetailRouting } from "@/app/(dashboard)/models-and-endpoints/detailNavigation";
import { useModelDashboardData } from "@/app/(dashboard)/models-and-endpoints/useModelDashboardData";
import AllModelsPanel from "@/app/(dashboard)/models-and-endpoints/panels/AllModelsPanel";
import AutoRoutersTabPanel from "@/app/(dashboard)/models-and-endpoints/panels/AutoRoutersTabPanel";
import AddModelPanel from "@/app/(dashboard)/models-and-endpoints/panels/AddModelPanel";
import LlmCredentialsPanel from "@/app/(dashboard)/models-and-endpoints/panels/LlmCredentialsPanel";
import PassThroughPanel from "@/app/(dashboard)/models-and-endpoints/panels/PassThroughPanel";
import HealthStatusPanel from "@/app/(dashboard)/models-and-endpoints/panels/HealthStatusPanel";
import ModelRetrySettingsPanel from "@/app/(dashboard)/models-and-endpoints/panels/ModelRetrySettingsPanel";
import ModelGroupAliasPanel from "@/app/(dashboard)/models-and-endpoints/panels/ModelGroupAliasPanel";
import PriceDataPanel from "@/app/(dashboard)/models-and-endpoints/panels/PriceDataPanel";

type ModelTabSlug =
  | "add"
  | "auto-routers"
  | "llm-credentials"
  | "pass-through"
  | "health"
  | "retry-settings"
  | "model-group-alias"
  | "price-data";

const BASE_TAB_KEY = "all-models";

const TAB_LABELS: Record<ModelTabSlug, string> = {
  add: "Add Model",
  "auto-routers": "Auto-Routers",
  "llm-credentials": "LLM Credentials",
  "pass-through": "Pass-Through Endpoints",
  health: "Health Status",
  "retry-settings": "Model Retry Settings",
  "model-group-alias": "Model Group Alias",
  "price-data": "Price Data Reload",
};

const renderPanel = (key: string) => {
  switch (key) {
    case BASE_TAB_KEY:
      return <AllModelsPanel />;
    case "auto-routers":
      return <AutoRoutersTabPanel />;
    case "add":
      return <AddModelPanel />;
    case "llm-credentials":
      return <LlmCredentialsPanel />;
    case "pass-through":
      return <PassThroughPanel />;
    case "health":
      return <HealthStatusPanel />;
    case "retry-settings":
      return <ModelRetrySettingsPanel />;
    case "model-group-alias":
      return <ModelGroupAliasPanel />;
    case "price-data":
      return <PriceDataPanel />;
    default:
      return null;
  }
};

export default function ModelsAndEndpointsPage() {
  const { accessToken, userRole, userId: userID, premiumUser } = useAuthorized();
  const { data: teams } = useTeams();
  const { data: uiSettings } = useUISettings();
  const queryClient = useQueryClient();
  const { modelId, teamId, close } = useModelDetailRouting();
  const { availableModelAccessGroups, allModelsOnProxy } = useModelDashboardData();

  const [activeKey, setActiveKey] = useState<string>(BASE_TAB_KEY);
  const [lastRefreshed, setLastRefreshed] = useState("");

  const isInternalUser = userRole && internalUserRoles.includes(userRole);
  const canCreate = canCreateModels(
    { userRole, userID },
    {
      teams: teams ?? null,
      disabledForInternalUsers:
        isInternalUser === true && uiSettings?.values?.disable_model_add_for_internal_users === true,
    },
  );
  const isAdmin = all_admin_roles.includes(userRole);

  const visibleSlugs = useMemo<Array<"" | ModelTabSlug>>(
    () => [
      "",
      ...(canCreate ? (["add"] as const) : []),
      ...(isAdmin || canCreate ? (["auto-routers"] as const) : []),
      ...(isAdmin
        ? (["llm-credentials", "pass-through", "health", "retry-settings", "model-group-alias", "price-data"] as const)
        : []),
    ],
    [canCreate, isAdmin],
  );

  const allModelsLabel = isAdmin ? "All Models" : "Your Models";
  // Auto-Routers carries a Beta badge; BetaBadge honours the admin setting that hides these.
  const tabLabel = (slug: "" | ModelTabSlug): React.ReactNode => {
    if (!slug) return allModelsLabel;
    if (slug === "auto-routers") {
      return (
        <span className="flex items-center gap-2">
          {TAB_LABELS[slug]} <BetaBadge />
        </span>
      );
    }
    return TAB_LABELS[slug];
  };

  const tabItems = visibleSlugs.map((slug) => {
    const key = slug || BASE_TAB_KEY;
    return {
      key,
      label: tabLabel(slug),
      children: key === activeKey ? renderPanel(key) : null,
    };
  });

  const handleRefreshClick = () => {
    setLastRefreshed(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    queryClient.invalidateQueries({ queryKey: ["models", "list"] });
  };

  const invalidateModels = () => queryClient.invalidateQueries({ queryKey: ["models", "list"] });

  if (teamId) {
    return (
      <div className="w-full h-full">
        <TeamInfoView
          teamId={teamId}
          onClose={close}
          accessToken={accessToken}
          is_team_admin={userRole === "Admin"}
          is_proxy_admin={userRole === "Proxy Admin"}
          userModels={allModelsOnProxy}
          editTeam={false}
          onUpdate={invalidateModels}
          premiumUser={premiumUser}
        />
      </div>
    );
  }

  return (
    <div className="mx-4 h-[75vh]">
      <div className="flex flex-col gap-2 p-8 w-full mt-2">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-lg font-semibold">Model Management</h2>
            {isAdmin ? (
              <p className="text-sm text-gray-600">Add and manage models for the proxy</p>
            ) : (
              <p className="text-sm text-gray-600">Add models for teams you are an admin for.</p>
            )}
          </div>
        </div>

        <CostOptimizationFeedbackBanner />

        {modelId ? (
          <ModelInfoView
            modelId={modelId}
            onClose={close}
            accessToken={accessToken}
            userID={userID}
            userRole={userRole}
            onModelUpdate={invalidateModels}
            modelAccessGroups={availableModelAccessGroups}
          />
        ) : (
          <Tabs
            activeKey={activeKey}
            onChange={setActiveKey}
            items={tabItems}
            tabBarExtraContent={{
              right: (
                <div className="flex items-center space-x-2 self-center">
                  {lastRefreshed && <span className="text-xs text-gray-500">Last Refreshed: {lastRefreshed}</span>}
                  <button
                    type="button"
                    onClick={handleRefreshClick}
                    aria-label="Refresh models"
                    className="cursor-pointer"
                  >
                    <RefreshIcon className="h-4 w-4 text-gray-500" />
                  </button>
                </div>
              ),
            }}
          />
        )}
      </div>
    </div>
  );
}
