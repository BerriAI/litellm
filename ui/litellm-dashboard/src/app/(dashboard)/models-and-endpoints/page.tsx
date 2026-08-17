"use client";

import { useMemo, useState } from "react";
import { RefreshCw } from "lucide-react";
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
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { useTranslation } from "react-i18next";

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
  const { t } = useTranslation();
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

  const tabLabels: Record<ModelTabSlug, string> = {
    add: t("models.addModel"),
    "auto-routers": t("models.autoRouters"),
    "llm-credentials": t("models.credentials"),
    "pass-through": t("models.passThrough"),
    health: t("models.health"),
    "retry-settings": t("models.retrySettings"),
    "model-group-alias": t("models.groupAlias"),
    "price-data": t("models.priceReload"),
  };
  const allModelsLabel = isAdmin ? t("models.allModels") : t("models.yourModels");
  const tabLabel = (slug: "" | ModelTabSlug): React.ReactNode => {
    if (!slug) return allModelsLabel;
    if (slug === "auto-routers") {
      return (
        <span className="flex items-center gap-2">
          {tabLabels[slug]} <BetaBadge />
        </span>
      );
    }
    return tabLabels[slug];
  };

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
    <div className="mx-4">
      <div className="mt-2 flex w-full flex-col gap-2 p-8">
        <div className="mb-4 flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold">{t("models.title")}</h2>
            {isAdmin ? (
              <p className="text-sm text-muted-foreground">{t("models.adminSubtitle")}</p>
            ) : (
              <p className="text-sm text-muted-foreground">{t("models.userSubtitle")}</p>
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
          <Tabs value={activeKey} onValueChange={setActiveKey}>
            <div className="flex min-w-0 flex-nowrap items-center gap-3 border-b">
              <div className="no-scrollbar scroll-fade-e -mb-1.5 min-w-0 flex-1 overflow-x-auto pb-1.5">
                <TabsList variant="line" className="w-max justify-start">
                  {visibleSlugs.map((slug) => {
                    const key = slug || BASE_TAB_KEY;
                    return (
                      <TabsTrigger key={key} value={key} className="flex-none">
                        {tabLabel(slug)}
                      </TabsTrigger>
                    );
                  })}
                </TabsList>
              </div>
              <div className="flex shrink-0 items-center gap-2 pb-1">
                {lastRefreshed && (
                  <span className="text-xs text-muted-foreground">
                    {t("models.lastRefreshed", { time: lastRefreshed })}
                  </span>
                )}
                <Button variant="ghost" size="icon-sm" onClick={handleRefreshClick} aria-label={t("models.refresh")}>
                  <RefreshCw />
                </Button>
              </div>
            </div>
            {visibleSlugs.map((slug) => {
              const key = slug || BASE_TAB_KEY;
              return (
                <TabsContent key={key} value={key} className="pt-4">
                  {renderPanel(key)}
                </TabsContent>
              );
            })}
          </Tabs>
        )}
      </div>
    </div>
  );
}
