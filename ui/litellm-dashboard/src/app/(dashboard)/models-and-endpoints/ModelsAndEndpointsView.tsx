import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useModelsInfo } from "@/app/(dashboard)/hooks/models/useModels";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { useUpdateRetryPolicy } from "@/app/(dashboard)/hooks/routerSettings/useUpdateRetryPolicy";
import AllModelsTab from "@/app/(dashboard)/models-and-endpoints/components/AllModelsTab";
import CostOptimizationFeedbackBanner from "@/components/molecules/cost_optimization_feedback_banner";
import ModelRetrySettingsTab from "@/app/(dashboard)/models-and-endpoints/components/ModelRetrySettingsTab";
import PriceDataManagementTab from "@/app/(dashboard)/models-and-endpoints/components/PriceDataManagementTab";
import { handleAddModelSubmit } from "@/components/add_model/handle_add_model_submit";
import { Team } from "@/components/key_team_helpers/key_list";
import CredentialsPanel from "@/components/model_add/CredentialsPanel";
import { getCallbacksCall } from "@/components/networking";
import { Providers, getPlaceholder, getProviderModels } from "@/components/provider_info_helpers";
import { getDisplayModelName } from "@/components/view_model/model_name_display";
import { transformModelData } from "./utils/modelDataTransformer";
import { all_admin_roles, internalUserRoles, isProxyAdminRole, isUserTeamAdminForAnyTeam } from "@/utils/roles";
import { RefreshIcon } from "@heroicons/react/outline";
import { useQueryClient } from "@tanstack/react-query";
import type { PaginationState } from "@tanstack/react-table";
import { Col, Grid, Icon, Tab, TabGroup, TabList, TabPanel, TabPanels } from "@tremor/react";
import type { UploadProps } from "antd";
import { Form } from "antd";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import AddModelTab from "../../../components/add_model/add_model_tab";
import HealthCheckComponent from "../../../components/model_dashboard/HealthCheckComponent";
import ModelGroupAliasSettings from "../../../components/model_group_alias_settings";
import ModelInfoView from "../../../components/model_info_view";
import NotificationsManager from "../../../components/molecules/notifications_manager";
import PassThroughSettings from "../../../components/PassThroughSettings/PassThroughSettings";
import TeamInfoView from "../../../components/team/TeamInfo";
import useAuthorized from "../hooks/useAuthorized";

interface ModelDashboardProps {
  premiumUser: boolean;
  teams: Team[] | null;
}

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
  model_group_alias?: { [key: string]: string } | null;
}

const HEALTH_PAGE_SIZE = 50;

const ModelsAndEndpointsView: React.FC<ModelDashboardProps> = ({ premiumUser, teams }) => {
  const { accessToken, token, userRole, userId: userID } = useAuthorized();
  const [addModelForm] = Form.useForm();
  const [lastRefreshed, setLastRefreshed] = useState("");
  const [providerModels, setProviderModels] = useState<Array<string>>([]);
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.Anthropic);
  const [selectedModelGroup, setSelectedModelGroup] = useState<string | null>(null);

  const [retryScope, setRetryScope] = useState<string | null>("global");
  const [modelGroupRetryPolicy, setModelGroupRetryPolicy] = useState<RetryPolicyObject | null>(null);
  const [globalRetryPolicy, setGlobalRetryPolicy] = useState<GlobalRetryPolicyObject | null>(null);
  const [defaultRetry, setDefaultRetry] = useState<number>(0);
  const [modelGroupAlias, setModelGroupAlias] = useState<{ [key: string]: string }>({});
  const [showAdvancedSettings, setShowAdvancedSettings] = useState<boolean>(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedTeamId, setSelectedTeamId] = useState<string | null>(null);
  const [selectedTabIndex, setSelectedTabIndex] = useState(0);
  const [healthPagination, setHealthPagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize: HEALTH_PAGE_SIZE,
  });

  const queryClient = useQueryClient();
  const { data: modelDataResponse, isLoading: isLoadingModels, refetch: refetchModels } = useModelsInfo();
  const { data: healthModelDataResponse, isLoading: isLoadingHealthModels } = useModelsInfo(
    healthPagination.pageIndex + 1,
    healthPagination.pageSize,
  );
  const { data: modelCostMapData, isLoading: isLoadingModelCostMap } = useModelCostMap();
  const { data: credentialsResponse, isLoading: isLoadingCredentials } = useCredentials();
  const credentialsList = credentialsResponse?.credentials || [];
  const { data: uiSettings, isLoading: isLoadingUISettings } = useUISettings();
  const updateRetryPolicy = useUpdateRetryPolicy(accessToken);

  const availableModelGroups = useMemo(() => {
    if (!modelDataResponse?.data) return [];
    const allModelGroups = new Set<string>();
    for (const model of modelDataResponse.data) {
      allModelGroups.add(model.model_name);
    }
    return Array.from(allModelGroups).sort();
  }, [modelDataResponse?.data]);

  const availableModelAccessGroups = useMemo(() => {
    if (!modelDataResponse?.data) return [];
    const allModelAccessGroups = new Set<string>();
    for (const model of modelDataResponse.data) {
      const modelInfo = model.model_info;
      if (modelInfo?.access_groups) {
        for (const group of modelInfo.access_groups) {
          allModelAccessGroups.add(group);
        }
      }
    }
    return Array.from(allModelAccessGroups);
  }, [modelDataResponse?.data]);

  const allModelsOnProxy = useMemo<string[]>(() => {
    if (!modelDataResponse?.data) return [];
    return modelDataResponse.data.map((model: any) => model.model_name);
  }, [modelDataResponse?.data]);

  const healthModelIdsOnProxy = useMemo<string[]>(() => {
    if (!healthModelDataResponse?.data) return [];
    return healthModelDataResponse.data
      .map((model: any) => model.model_info?.id)
      .filter((id: string | undefined): id is string => Boolean(id));
  }, [healthModelDataResponse?.data]);

  const getProviderFromModel = (model: string) => {
    if (modelCostMapData !== null && modelCostMapData !== undefined) {
      if (typeof modelCostMapData == "object" && model in modelCostMapData) {
        return modelCostMapData[model]["litellm_provider"];
      }
    }
    return "openai";
  };

  const processedModelData = useMemo(() => {
    if (!modelDataResponse?.data) return { data: [] };
    return transformModelData(modelDataResponse, getProviderFromModel);
  }, [modelDataResponse?.data, getProviderFromModel]);

  const processedHealthModelData = useMemo(() => {
    if (!healthModelDataResponse?.data) return { data: [] };
    return transformModelData(healthModelDataResponse, getProviderFromModel);
  }, [healthModelDataResponse?.data, getProviderFromModel]);

  const healthRowCount = healthModelDataResponse?.total_count ?? 0;

  const isProxyAdmin = userRole && isProxyAdminRole(userRole);
  const isInternalUser = userRole && internalUserRoles.includes(userRole);
  const isUserTeamAdmin = userID && isUserTeamAdminForAnyTeam(teams, userID);
  const addModelDisabledForInternalUsers =
    isInternalUser && uiSettings?.values?.disable_model_add_for_internal_users === true;
  // Hide tab if user is NOT a proxy admin AND (internal user with setting enabled OR not a team admin)
  const shouldHideAddModelTab = !isProxyAdmin && (addModelDisabledForInternalUsers || !isUserTeamAdmin);

  const setProviderModelsFn = (provider: Providers) => {
    const _providerModels = getProviderModels(provider, modelCostMapData);
    setProviderModels(_providerModels);
  };

  const uploadProps: UploadProps = {
    name: "file",
    accept: ".json",
    pastable: false,
    beforeUpload: (file) => {
      if (file.type === "application/json") {
        const reader = new FileReader();
        reader.onload = (e) => {
          if (e.target) {
            const jsonStr = e.target.result as string;
            addModelForm.setFieldsValue({ vertex_credentials: jsonStr });
          }
        };
        reader.readAsText(file);
      }
      return false;
    },
    onChange(info) {
      if (info.file.status === "done") {
        NotificationsManager.success(`${info.file.name} file uploaded successfully`);
      } else if (info.file.status === "error") {
        NotificationsManager.fromBackend(`${info.file.name} file upload failed.`);
      }
    },
  };

  const handleRefreshClick = () => {
    const currentDate = new Date();
    setLastRefreshed(currentDate.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    setHealthPagination((previous) => ({ ...previous, pageIndex: 0 }));
    queryClient.invalidateQueries({ queryKey: ["models", "list"] });
    refetchModels();
  };

  const fetchRouterSettings = useCallback(async (): Promise<RouterSettings | null> => {
    if (!accessToken || !userID || !userRole) {
      return null;
    }
    try {
      const routerSettingsInfo = await getCallbacksCall(accessToken, userID, userRole);
      return routerSettingsInfo.router_settings;
    } catch (error) {
      console.error("Error fetching model data:", error);
      return null;
    }
  }, [accessToken, userID, userRole]);

  const applyRouterSettings = useCallback((routerSettings: RouterSettings) => {
    setModelGroupRetryPolicy(routerSettings.model_group_retry_policy ?? null);
    setGlobalRetryPolicy(routerSettings.retry_policy ?? null);
    setDefaultRetry(routerSettings.num_retries ?? 2);
    setModelGroupAlias(routerSettings.model_group_alias || {});
  }, []);

  const loadRetrySettings = useCallback(async () => {
    const routerSettings = await fetchRouterSettings();
    if (routerSettings) {
      applyRouterSettings(routerSettings);
    }
  }, [fetchRouterSettings, applyRouterSettings]);

  const handleSaveRetrySettings = () => {
    updateRetryPolicy.mutate(
      {
        retry_policy: globalRetryPolicy,
        model_group_retry_policy: modelGroupRetryPolicy,
      },
      {
        onSuccess: () => {
          NotificationsManager.success("Retry settings saved successfully");
          loadRetrySettings();
        },
        onError: () => {
          NotificationsManager.fromBackend("Failed to save retry settings");
        },
      },
    );
  };

  useEffect(() => {
    if (!accessToken || !token || !userRole || !userID || !modelDataResponse) {
      return;
    }
    let active = true;
    void (async () => {
      const routerSettings = await fetchRouterSettings();
      if (active && routerSettings) {
        applyRouterSettings(routerSettings);
      }
    })();
    return () => {
      active = false;
    };
  }, [accessToken, token, userRole, userID, modelDataResponse, fetchRouterSettings, applyRouterSettings]);

  const isLoading = isLoadingModels || isLoadingModelCostMap || isLoadingCredentials || isLoadingUISettings;

  // Admin Viewer can view all models read-only — page render proceeds; the
  // individual write-action tabs (Add Model, LLM Credentials, etc.) are
  // gated separately below.

  const handleOk = async () => {
    try {
      const values = await addModelForm.validateFields();
      await handleAddModelSubmit(values, accessToken, addModelForm, handleRefreshClick);
    } catch (error: any) {
      const errorMessages =
        error.errorFields
          ?.map((field: any) => {
            return `${field.name.join(".")}: ${field.errors.join(", ")}`;
          })
          .join(" | ") || "Unknown validation error";
      NotificationsManager.fromBackend(`Please fill in the following required fields: ${errorMessages}`);
    }
  };

  Object.keys(Providers).find((key) => (Providers as { [index: string]: any })[key] === selectedProvider);
  // If a team is selected, render TeamInfoView in full page layout
  if (selectedTeamId) {
    return (
      <div className="w-full h-full">
        <TeamInfoView
          teamId={selectedTeamId}
          onClose={() => setSelectedTeamId(null)}
          accessToken={accessToken}
          is_team_admin={userRole === "Admin"}
          is_proxy_admin={userRole === "Proxy Admin"}
          userModels={allModelsOnProxy}
          editTeam={false}
          onUpdate={handleRefreshClick}
          premiumUser={premiumUser}
        />
      </div>
    );
  }

  return (
    <div className="mx-4 h-[75vh]">
      <Grid numItems={1} className="gap-2 p-8 w-full mt-2">
        <Col numColSpan={1} className="flex flex-col gap-2">
          {/* Model Management Header */}
          <div className="flex justify-between items-center mb-4">
            <div>
              <h2 className="text-lg font-semibold">Model Management</h2>
              {!all_admin_roles.includes(userRole) ? (
                <p className="text-sm text-gray-600">Add models for teams you are an admin for.</p>
              ) : (
                <p className="text-sm text-gray-600">Add and manage models for the proxy</p>
              )}
            </div>
          </div>

          {/* Cost Optimization Feedback Banner */}
          <CostOptimizationFeedbackBanner />
          {selectedModelId && !isLoading ? (
            <ModelInfoView
              modelId={selectedModelId}
              onClose={() => {
                setSelectedModelId(null);
              }}
              accessToken={accessToken}
              userID={userID}
              userRole={userRole}
              onModelUpdate={(updatedModel) => {
                queryClient.invalidateQueries({ queryKey: ["models", "list"] });
                handleRefreshClick();
              }}
              modelAccessGroups={availableModelAccessGroups}
            />
          ) : (
            (() => {
              // Build a single source-of-truth list of {tab, panel} pairs.
              // Conditionally-hidden tabs (e.g. "Add Model" for non-admin) get
              // filtered out as a unit so tab indices and panel indices can
              // never drift apart — Tremor's TabList and TabPanels filter
              // falsy children inconsistently, which previously caused
              // "click LLM Credentials, see nothing" for Admin Viewer.
              const isAdmin = all_admin_roles.includes(userRole);
              const visibleTabs: Array<{ tab: React.ReactElement; panel: React.ReactElement }> = [
                {
                  tab: <Tab key="all-models">{isAdmin ? "All Models" : "Your Models"}</Tab>,
                  panel: (
                    <AllModelsTab
                      key="all-models"
                      selectedModelGroup={selectedModelGroup}
                      setSelectedModelGroup={setSelectedModelGroup}
                      availableModelGroups={availableModelGroups}
                      availableModelAccessGroups={availableModelAccessGroups}
                      setSelectedModelId={setSelectedModelId}
                      setSelectedTeamId={setSelectedTeamId}
                    />
                  ),
                },
              ];
              if (!shouldHideAddModelTab) {
                visibleTabs.push({
                  tab: <Tab key="add-model">Add Model</Tab>,
                  panel: (
                    <TabPanel key="add-model" className="h-full">
                      <AddModelTab
                        form={addModelForm}
                        handleOk={handleOk}
                        selectedProvider={selectedProvider}
                        setSelectedProvider={setSelectedProvider}
                        providerModels={providerModels}
                        setProviderModelsFn={setProviderModelsFn}
                        getPlaceholder={getPlaceholder}
                        uploadProps={uploadProps}
                        showAdvancedSettings={showAdvancedSettings}
                        setShowAdvancedSettings={setShowAdvancedSettings}
                        teams={teams}
                        credentials={credentialsList}
                        accessToken={accessToken}
                        userRole={userRole}
                      />
                    </TabPanel>
                  ),
                });
              }
              if (isAdmin) {
                visibleTabs.push(
                  {
                    tab: <Tab key="llm-credentials">LLM Credentials</Tab>,
                    panel: (
                      <TabPanel key="llm-credentials">
                        <CredentialsPanel uploadProps={uploadProps} />
                      </TabPanel>
                    ),
                  },
                  {
                    tab: <Tab key="pass-through">Pass-Through Endpoints</Tab>,
                    panel: (
                      <TabPanel key="pass-through">
                        <PassThroughSettings
                          accessToken={accessToken}
                          userRole={userRole}
                          userID={userID}
                          premiumUser={premiumUser}
                        />
                      </TabPanel>
                    ),
                  },
                  {
                    tab: <Tab key="health-status">Health Status</Tab>,
                    panel: (
                      <TabPanel key="health-status">
                        <HealthCheckComponent
                          accessToken={accessToken}
                          modelData={processedHealthModelData}
                          all_models_on_proxy={healthModelIdsOnProxy}
                          getDisplayModelName={getDisplayModelName}
                          setSelectedModelId={setSelectedModelId}
                          teams={teams}
                          isLoading={isLoadingHealthModels}
                          pagination={healthPagination}
                          onPaginationChange={setHealthPagination}
                          rowCount={healthRowCount}
                        />
                      </TabPanel>
                    ),
                  },
                  {
                    tab: <Tab key="model-retry-settings">Model Retry Settings</Tab>,
                    panel: (
                      <ModelRetrySettingsTab
                        key="model-retry-settings"
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
                    ),
                  },
                  {
                    tab: <Tab key="model-group-alias">Model Group Alias</Tab>,
                    panel: (
                      <TabPanel key="model-group-alias">
                        <ModelGroupAliasSettings
                          accessToken={accessToken}
                          initialModelGroupAlias={modelGroupAlias}
                          onAliasUpdate={setModelGroupAlias}
                        />
                      </TabPanel>
                    ),
                  },
                  {
                    tab: <Tab key="price-data-reload">Price Data Reload</Tab>,
                    panel: <PriceDataManagementTab key="price-data-reload" />,
                  },
                );
              }
              return (
                <TabGroup
                  index={selectedTabIndex}
                  onIndexChange={setSelectedTabIndex}
                  className="gap-2 h-[75vh] w-full "
                >
                  <TabList className="flex justify-between mt-2 w-full items-center">
                    <div className="flex">{visibleTabs.map((t) => t.tab)}</div>

                    <div className="flex items-center space-x-2 self-center">
                      {lastRefreshed && <span className="text-xs text-gray-500">Last Refreshed: {lastRefreshed}</span>}
                      <Icon
                        icon={RefreshIcon}
                        variant="shadow"
                        size="xs"
                        className="cursor-pointer"
                        onClick={handleRefreshClick}
                      />
                    </div>
                  </TabList>
                  <TabPanels>{visibleTabs.map((t) => t.panel)}</TabPanels>
                </TabGroup>
              );
            })()
          )}
        </Col>
      </Grid>
    </div>
  );
};

export default ModelsAndEndpointsView;
