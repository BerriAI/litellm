import { useCredentials } from "@/app/(dashboard)/hooks/credentials/useCredentials";
import { useModelsInfo } from "@/app/(dashboard)/hooks/models/useModels";
import { useModelCostMap } from "@/app/(dashboard)/hooks/models/useModelCostMap";
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
import { all_admin_roles, internalUserRoles, isProxyAdminRole, isUserTeamAdminForAnyTeam } from "@/utils/roles";
import { RefreshIcon } from "@heroicons/react/outline";
import { useQueryClient } from "@tanstack/react-query";
import { Col, Grid, Icon, Tab, TabGroup, TabList, TabPanel, TabPanels, Text } from "@tremor/react";
import type { UploadProps } from "antd";
import { Form, Typography } from "antd";
import React, { useEffect, useRef, useState } from "react";
import AddModelTab from "../../../components/add_model/add_model_tab";
import HealthCheckComponent from "../../../components/model_dashboard/HealthCheckComponent";
import ModelGroupAliasSettings from "../../../components/model_group_alias_settings";
import ModelInfoView from "../../../components/model_info_view";
import NotificationsManager from "../../../components/molecules/notifications_manager";
import PassThroughSettings from "../../../components/pass_through_settings";
import TeamInfoView from "../../../components/team/team_info";
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

interface GlobalExceptionActivityData {
  sum_num_rate_limit_exceptions: number;
  daily_data: { date: string; num_rate_limit_exceptions: number }[];
}

//["OpenAI", "Azure OpenAI", "Anthropic", "Gemini (Google AI Studio)", "Amazon Bedrock", "OpenAI-Compatible Endpoints (Groq, Together AI, Mistral AI, etc.)"]

interface ProviderFields {
  field_name: string;
  field_type: string;
  field_description: string;
  field_value: string;
}

interface ProviderSettings {
  name: string;
  fields: ProviderFields[];
}

const ModelsAndEndpointsView: React.FC<ModelDashboardProps> = ({
  modelData = { data: [] },
  keys,
  setModelData,
  premiumUser,
  teams,
}) => {
  const { accessToken, token, userRole, userId: userID } = useAuthorized();
  const [addModelForm] = Form.useForm();
  const [lastRefreshed, setLastRefreshed] = useState("");
  const [providerModels, setProviderModels] = useState<Array<string>>([]); // Explicitly typing providerModels as a string array
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.Anthropic);
  const [availableModelGroups, setAvailableModelGroups] = useState<Array<string>>([]);
  const [availableModelAccessGroups, setAvailableModelAccessGroups] = useState<Array<string>>([]);
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
  const { data: modelCostMapData } = useModelCostMap();
  const { data: credentialsResponse } = useCredentials();
  const credentialsList = credentialsResponse?.credentials || [];
  const { data: uiSettings } = useUISettings(accessToken || "");

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
        setModelData(modelDataResponse);
        let all_model_groups: Set<string> = new Set();
        for (let i = 0; i < modelDataResponse.data.length; i++) {
          const model = modelDataResponse.data[i];
          all_model_groups.add(model.model_name);
        }
        let _array_model_groups = Array.from(all_model_groups);
        _array_model_groups = _array_model_groups.sort();

        setAvailableModelGroups(_array_model_groups);

        let all_model_access_groups: Set<string> = new Set();
        for (let i = 0; i < modelDataResponse.data.length; i++) {
          const model = modelDataResponse.data[i];
          let model_info: any | null = model.model_info;
          if (model_info) {
            let access_groups = model_info.access_groups;
            if (access_groups) {
              for (let j = 0; j < access_groups.length; j++) {
                all_model_access_groups.add(access_groups[j]);
              }
            }
          }
        }

        setAvailableModelAccessGroups(Array.from(all_model_access_groups));

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

  if (!modelData || isLoadingModels) {
    return <div>Loading...</div>;
  }

  let all_models_on_proxy: any[] = [];
  let all_providers: string[] = [];

  // loop through model data and edit each row
  for (let i = 0; i < modelData.data.length; i++) {
    let curr_model = modelData.data[i];
    let litellm_model_name = curr_model?.litellm_params?.model;
    let custom_llm_provider = curr_model?.litellm_params?.custom_llm_provider;
    let model_info = curr_model?.model_info;

    let provider = "";
    let input_cost = "Undefined";
    let output_cost = "Undefined";
    let max_tokens = "Undefined";
    let max_input_tokens = "Undefined";
    let cleanedLitellmParams = {};

    const getProviderFromModel = (model: string) => {
      /**
       * Use model map
       * - check if model in model map
       * - return it's litellm_provider, if so
       */
      if (modelCostMapData !== null && modelCostMapData !== undefined) {
        if (typeof modelCostMapData == "object" && model in modelCostMapData) {
          return modelCostMapData[model]["litellm_provider"];
        }
      }
      return "openai";
    };

    // Check if litellm_model_name is null or undefined
    if (litellm_model_name) {
      // Split litellm_model_name based on "/"
      let splitModel = litellm_model_name.split("/");

      // Get the first element in the split
      let firstElement = splitModel[0];

      // If there is only one element, default provider to openai
      provider = custom_llm_provider;
      if (!provider) {
        provider = splitModel.length === 1 ? getProviderFromModel(litellm_model_name) : firstElement;
      }
    } else {
      // litellm_model_name is null or undefined, default provider to openai
      provider = "-";
    }

    if (model_info) {
      input_cost = model_info?.input_cost_per_token;
      output_cost = model_info?.output_cost_per_token;
      max_tokens = model_info?.max_tokens;
      max_input_tokens = model_info?.max_input_tokens;
    }

    if (curr_model?.litellm_params) {
      cleanedLitellmParams = Object.fromEntries(
        Object.entries(curr_model?.litellm_params).filter(([key]) => key !== "model" && key !== "api_base"),
      );
    }

    modelData.data[i].provider = provider;
    modelData.data[i].input_cost = input_cost;
    modelData.data[i].output_cost = output_cost;
    modelData.data[i].litellm_model_name = litellm_model_name;
    all_providers.push(provider);

    // Convert Cost in terms of Cost per 1M tokens
    if (modelData.data[i].input_cost) {
      modelData.data[i].input_cost = (Number(modelData.data[i].input_cost) * 1000000).toFixed(2);
    }

    if (modelData.data[i].output_cost) {
      modelData.data[i].output_cost = (Number(modelData.data[i].output_cost) * 1000000).toFixed(2);
    }

    modelData.data[i].max_tokens = max_tokens;
    modelData.data[i].max_input_tokens = max_input_tokens;
    modelData.data[i].api_base = curr_model?.litellm_params?.api_base;
    modelData.data[i].cleanedLitellmParams = cleanedLitellmParams;

    all_models_on_proxy.push(curr_model.model_name);
  }
  // when users click request access show pop up to allow them to request access

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
          userModels={all_models_on_proxy}
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
          {selectedModelId ? (
            <ModelInfoView
              modelId={selectedModelId}
              onClose={() => {
                setSelectedModelId(null);
              }}
              modelData={modelData.data.find((model: any) => model.model_info.id === selectedModelId)}
              accessToken={accessToken}
              userID={userID}
              userRole={userRole}
              onModelUpdate={(updatedModel) => {
                // Handle model deletion
                if (updatedModel.deleted) {
                  const updatedModelData = {
                    ...modelData,
                    data: modelData.data.filter((model: any) => model.model_info.id !== updatedModel.model_info.id),
                  };
                  setModelData(updatedModelData);
                } else {
                  // Update the model in the modelData.data array
                  const updatedModelData = {
                    ...modelData,
                    data: modelData.data.map((model: any) =>
                      model.model_info.id === updatedModel.model_info.id ? updatedModel : model,
                    ),
                  };
                  setModelData(updatedModelData);
                }
                // Invalidate cache and trigger a refresh to update UI
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
                    modelData={modelData}
                    premiumUser={premiumUser}
                  />
                </TabPanel>
                <TabPanel>
                  <HealthCheckComponent
                    accessToken={accessToken}
                    modelData={modelData}
                    all_models_on_proxy={all_models_on_proxy}
                    getDisplayModelName={getDisplayModelName}
                    setSelectedModelId={setSelectedModelId}
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
