"use client";

import type { ReactNode } from "react";
import { useEffect, useMemo, useState } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Tabs } from "antd";
import { RefreshIcon } from "@heroicons/react/outline";
import { useQueryClient } from "@tanstack/react-query";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useTeams } from "@/app/(dashboard)/hooks/teams/useTeams";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { all_admin_roles, internalUserRoles, isProxyAdminRole, isUserTeamAdminForAnyTeam } from "@/utils/roles";
import CostOptimizationFeedbackBanner from "@/components/molecules/cost_optimization_feedback_banner";
import ModelInfoView from "@/components/model_info_view";
import TeamInfoView from "@/components/team/TeamInfo";
import { modelTabHref, slugFromPathname, type ModelTabSlug } from "@/app/(dashboard)/models-and-endpoints/tabRoutes";
import { useModelDetailRouting } from "@/app/(dashboard)/models-and-endpoints/detailNavigation";
import { useModelDashboardData } from "@/app/(dashboard)/models-and-endpoints/useModelDashboardData";

const BASE_TAB_KEY = "all-models";

const TAB_LABELS: Record<ModelTabSlug, string> = {
  add: "Add Model",
  "llm-credentials": "LLM Credentials",
  "pass-through": "Pass-Through Endpoints",
  health: "Health Status",
  "retry-settings": "Model Retry Settings",
  "model-group-alias": "Model Group Alias",
  "price-data": "Price Data Reload",
};

export default function ModelsAndEndpointsLayout({ children }: { children: ReactNode }) {
  const { accessToken, userRole, userId: userID, premiumUser } = useAuthorized();
  const { data: teams, isLoading: teamsLoading } = useTeams();
  const { data: uiSettings, isLoading: uiSettingsLoading } = useUISettings();
  const pathname = usePathname();
  const router = useRouter();
  const queryClient = useQueryClient();
  const { modelId, teamId, close } = useModelDetailRouting();
  const { availableModelAccessGroups, allModelsOnProxy } = useModelDashboardData();

  const [lastRefreshed, setLastRefreshed] = useState("");

  const isProxyAdmin = userRole && isProxyAdminRole(userRole);
  const isInternalUser = userRole && internalUserRoles.includes(userRole);
  const isUserTeamAdmin = userID && isUserTeamAdminForAnyTeam(teams ?? null, userID);
  const addModelDisabledForInternalUsers =
    isInternalUser && uiSettings?.values?.disable_model_add_for_internal_users === true;
  const shouldHideAddModelTab = !isProxyAdmin && (addModelDisabledForInternalUsers || !isUserTeamAdmin);
  const isAdmin = all_admin_roles.includes(userRole);

  const visibleSlugs = useMemo<Array<"" | ModelTabSlug>>(
    () => [
      "",
      ...(shouldHideAddModelTab ? [] : (["add"] as const)),
      ...(isAdmin
        ? (["llm-credentials", "pass-through", "health", "retry-settings", "model-group-alias", "price-data"] as const)
        : []),
    ],
    [shouldHideAddModelTab, isAdmin],
  );

  const activeSlug = slugFromPathname(pathname);
  const isKnownSlug = visibleSlugs.some((slug) => slug === activeSlug);
  const activeKey = isKnownSlug ? activeSlug || BASE_TAB_KEY : BASE_TAB_KEY;

  useEffect(() => {
    // Wait until the async inputs to tab visibility (teams, ui settings) have
    // loaded; otherwise a team admin hard-loading /add is redirected to the base
    // before their team membership resolves.
    if (teamsLoading || uiSettingsLoading) {
      return;
    }
    if (activeSlug !== "" && !isKnownSlug) {
      router.replace(modelTabHref(""));
    }
  }, [activeSlug, isKnownSlug, teamsLoading, uiSettingsLoading, router]);

  const allModelsLabel = isAdmin ? "All Models" : "Your Models";
  const tabItems = visibleSlugs.map((slug) => {
    const key = slug || BASE_TAB_KEY;
    return {
      key,
      label: slug ? TAB_LABELS[slug] : allModelsLabel,
      children: key === activeKey ? children : null,
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
            onChange={(key) => router.push(modelTabHref(key === BASE_TAB_KEY ? "" : key))}
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
