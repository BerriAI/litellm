import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
import { useModelsInfo } from "@/app/(dashboard)/hooks/models/useModels";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import AllModelsTab from "@/app/(dashboard)/models-and-endpoints/components/AllModelsTab";
import ModelRetrySettingsTab from "@/app/(dashboard)/models-and-endpoints/components/ModelRetrySettingsTab";
import PriceDataManagementTab from "@/app/(dashboard)/models-and-endpoints/components/PriceDataManagementTab";
import { handleAddModelSubmit } from "@/components/add_model/handle_add_model_submit";
import { Team } from "@/components/key_team_helpers/key_list";
import CredentialsPanel from "@/components/model_add/credentials";
import { getCallbacksCall, setCallbacksCall } from "@/components/networking";
import { Providers, getPlaceholder, getProviderModels } from "@/components/provider_info_helpers";
import { getDisplayModelName } from "@/components/view_model/model_name_display";
import { transformModelData } from "./utils/modelDataTransformer";
import { all_admin_roles, internalUserRoles, isProxyAdminRole, isUserTeamAdminForAnyTeam } from "@/utils/roles";
import { RefreshIcon } from "@heroicons/react/outline";
import { useQueryClient } from "@tanstack/react-query";
import { Col, Grid, Icon, Tab, TabGroup, TabList, TabPanel, TabPanels, Text } from "@tremor/react";
import type { UploadProps } from "antd";
import { Form, Typography } from "antd";
import { PlusCircleOutlined } from "@ant-design/icons";
import React, { useEffect, useMemo, useState } from "react";
import AddModelTab from "../../../components/add_model/add_model_tab";
import HealthCheckComponent from "../../../components/model_dashboard/HealthCheckComponent";
import ModelGroupAliasSettings from "../../../components/model_group_alias_settings";
import ModelInfoView from "../../../components/model_info_view";
import NotificationsManager from "../../../components/molecules/notifications_manager";
import PassThroughSettings from "../../../components/pass_through_settings";
import TeamInfoView from "../../../components/team/TeamInfo";
import useAuthorized from "../hooks/useAuthorized";

interface ModelDashboardProps {
  token: string | null;
  modelData: any;
  keys: any[] | null;
  setModelData: any;
  premiumUser: boolean;
  teams: Team[] | null;
}

interface RetryPolicyObject {
  [key: string]: { [retryPolicyKey: string]: number } | undefined;
}

interface GlobalRetryPolicyObject {
  [retryPolicyKey: string]: number;
}

const ModelsAndEndpointsView: React.FC<ModelDashboardProps> = ({ premiumUser, teams }) => {
  const { accessToken, token, userRole, userId: userID } = useAuthorized();
  const [addModelForm] = Form.useForm();
  const [lastRefreshed, setLastRefreshed] = useState("");
  const [providerModels, setProviderModels] = useState<Array<string>>([]);
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.Anthropic);
  const [selectedModelGroup, setSelectedModelGroup] = useState<string | null>(null);

  const [modelGroupRetryPolicy, setModelGroupRetryPolicy] = useState<RetryPolicyObject | null>(null);
  const [globalRetryPolicy, setGlobalRetryPolicy] = useState<GlobalRetryPolicyObject | null>(null);
  const [defaultRetry, setDefaultRetry] = useState<number>(0);
  const [modelGroupAlias, setModelGroupAlias] = useState<{ [key: string]: string }>({});
  const [showAdvancedSettings, setShowAdvancedSettings] = useState<boolean>(false);
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [selectedTeamId, setSelectedTeamId] = useState<string | null>(null);
  const [selectedTabIndex, setSelectedTabIndex] = useState(0);

  const queryClient = useQueryClient();
  const { data: modelDataResponse, isLoading: isLoadingModels, refetch: refetchModels } = useModelsInfo();
  const { data: modelCostMapData, isLoading: isLoadingModelCostMap } = useModelCostMap();
  const { data: credentialsResponse, isLoading: isLoadingCredentials } = useCredentials();
  const credentialsList = credentialsResponse?.credentials || [];
  const { data: uiSettings, isLoading: isLoadingUISettings } = useUISettings();

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
    setLastRefreshed(currentDate.toLocaleString());
    queryClient.invalidateQueries({ queryKey: ["models", "list"] });
    refetchModels();
  };

  const handleSaveRetrySettings = async () => {
    if (!accessToken) {
      return;
    }

    try {
      const payload: any = {
        router_settings: {},
      };

      if (selectedModelGroup === "global") {
        if (globalRetryPolicy) {
          payload.router_settings.retry_policy = globalRetryPolicy;
        }
        NotificationsManager.success("Global retry settings saved successfully");
      } else {
        if (modelGroupRetryPolicy) {
          payload.router_settings.model_group_retry_policy = modelGroupRetryPolicy;
        }
        NotificationsManager.success(`Retry settings saved successfully for ${selectedModelGroup}`);
      }

      await setCallbacksCall(accessToken, payload);
    } catch (error) {
      NotificationsManager.fromBackend("Failed to save retry settings");
    }
  };

  useEffect(() => {
    if (!accessToken || !token || !userRole || !userID || !modelDataResponse) {
      return;
    }
    const fetchData = async () => {
      try {
        const routerSettingsInfo = await getCallbacksCall(accessToken, userID, userRole);
        let router_settings = routerSettingsInfo.router_settings;

        let model_group_retry_policy = router_settings.model_group_retry_policy;
        let default_retries = router_settings.num_retries;

        setModelGroupRetryPolicy(model_group_retry_policy);
        setGlobalRetryPolicy(router_settings.retry_policy);
        setDefaultRetry(default_retries);

        const model_group_alias = router_settings.model_group_alias || {};
        setModelGroupAlias(model_group_alias);
      } catch (error) {
        console.error("Error fetching model data:", error);
      }
    };

    if (accessToken && token && userRole && userID && modelDataResponse) {
      fetchData();
    }
  }, [accessToken, token, userRole, userID, modelDataResponse]);

  const isLoading = isLoadingModels || isLoadingModelCostMap || isLoadingCredentials || isLoadingUISettings;

  if (userRole && userRole == "Admin Viewer") {
    const { Title, Paragraph } = Typography;
    return (
      <div>
        <Title level={1}>Access Denied</Title>
        <Paragraph>Ask your proxy admin for access to view all models</Paragraph>
      </div>
    );
  }

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
    <div className="w-full mx-4 h-[75vh]">
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

          {/* Missing Provider Banner */}
          <div className="mb-4 px-4 py-3 bg-blue-50 rounded-lg border border-blue-100 flex items-center gap-4">
            <div className="flex-shrink-0 w-10 h-10 bg-white rounded-full flex items-center justify-center border border-blue-200">
              <PlusCircleOutlined style={{ fontSize: "18px", color: "#6366f1" }} />
            </div>
            <div className="flex-1 min-w-0">
              <h4 className="text-gray-900 font-semibold text-sm m-0">Missing a provider?</h4>
              <p className="text-gray-500 text-xs m-0 mt-0.5">
                The LiteLLM engineering team is constantly adding support for new LLM models, providers, endpoints. If
                you don&apos;t see the one you need, let us know and we&apos;ll prioritize it.
              </p>
            </div>
            <a
              href="https://models.litellm.ai/?request=true"
              target="_blank"
              rel="noopener noreferrer"
              className="flex-shrink-0 inline-flex items-center gap-2 px-4 py-2 bg-[#6366f1] hover:bg-[#5558e3] text-white text-sm font-medium rounded-lg transition-colors"
            >
              Request Provider
              <svg
                xmlns="http://www.w3.org/2000/svg"
                className="h-4 w-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14"
                />
              </svg>
            </a>
          </div>
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
            <TabGroup index={selectedTabIndex} onIndexChange={setSelectedTabIndex} className="gap-2 h-[75vh] w-full ">
              <TabList className="flex justify-between mt-2 w-full items-center">
                <div className="flex">
                  {all_admin_roles.includes(userRole) ? <Tab>All Models</Tab> : <Tab>Your Models</Tab>}
                  {!shouldHideAddModelTab && <Tab>Add Model</Tab>}
                  {all_admin_roles.includes(userRole) && <Tab>LLM Credentials</Tab>}
                  {all_admin_roles.includes(userRole) && <Tab>Pass-Through Endpoints</Tab>}
                  {all_admin_roles.includes(userRole) && <Tab>Health Status</Tab>}
                  {all_admin_roles.includes(userRole) && <Tab>Model Retry Settings</Tab>}
                  {all_admin_roles.includes(userRole) && <Tab>Model Group Alias</Tab>}
                  {all_admin_roles.includes(userRole) && <Tab>Price Data Reload</Tab>}
                </div>

                <div className="flex items-center space-x-2">
                  {lastRefreshed && <Text>Last Refreshed: {lastRefreshed}</Text>}
                  <Icon
                    icon={RefreshIcon} // Modify as necessary for correct icon name
                    variant="shadow"
                    size="xs"
                    className="self-center"
                    onClick={handleRefreshClick}
                  />
                </div>
              </TabList>
              <TabPanels>
                <AllModelsTab
                  selectedModelGroup={selectedModelGroup}
                  setSelectedModelGroup={setSelectedModelGroup}
                  availableModelGroups={availableModelGroups}
                  availableModelAccessGroups={availableModelAccessGroups}
                  setSelectedModelId={setSelectedModelId}
                  setSelectedTeamId={setSelectedTeamId}
                />
                {!shouldHideAddModelTab && (
                  <TabPanel className="h-full">
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
                )}
                <TabPanel>
                  <CredentialsPanel uploadProps={uploadProps} />
                </TabPanel>
                <TabPanel>
                  <PassThroughSettings
                    accessToken={accessToken}
                    userRole={userRole}
                    userID={userID}
                    modelData={processedModelData}
                    premiumUser={premiumUser}
                  />
                </TabPanel>
                <TabPanel>
                  <HealthCheckComponent
                    accessToken={accessToken}
                    modelData={processedModelData}
                    all_models_on_proxy={allModelsOnProxy}
                    getDisplayModelName={getDisplayModelName}
                    setSelectedModelId={setSelectedModelId}
                    teams={teams}
                  />
                </TabPanel>
                <ModelRetrySettingsTab
                  selectedModelGroup={selectedModelGroup}
                  setSelectedModelGroup={setSelectedModelGroup}
                  availableModelGroups={availableModelGroups}
                  globalRetryPolicy={globalRetryPolicy}
                  setGlobalRetryPolicy={setGlobalRetryPolicy}
                  defaultRetry={defaultRetry}
                  modelGroupRetryPolicy={modelGroupRetryPolicy}
                  setModelGroupRetryPolicy={setModelGroupRetryPolicy}
                  handleSaveRetrySettings={handleSaveRetrySettings}
                />
                <TabPanel>
                  <ModelGroupAliasSettings
                    accessToken={accessToken}
                    initialModelGroupAlias={modelGroupAlias}
                    onAliasUpdate={setModelGroupAlias}
                  />
                </TabPanel>
                <PriceDataManagementTab />
              </TabPanels>
            </TabGroup>
          )}
        </Col>
      </Grid>
    </div>
  );
};

export default ModelsAndEndpointsView;
