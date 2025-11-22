import React, { useState, useEffect, useRef } from "react";
import { Text, Grid, Col } from "@tremor/react";
import { CredentialItem, credentialListCall, CredentialsResponse } from "@/components/networking";

import { handleAddModelSubmit } from "@/components/add_model/handle_add_model_submit";

import CredentialsPanel from "@/components/model_add/credentials";
import { getDisplayModelName } from "@/components/view_model/model_name_display";
import { TabPanel, TabPanels, TabGroup, TabList, Tab, Icon } from "@tremor/react";
import { DateRangePickerValue } from "@tremor/react";
import {
  modelInfoCall,
  modelCostMap,
  modelMetricsCall,
  streamingModelMetricsCall,
  modelExceptionsCall,
  modelMetricsSlowResponsesCall,
  getCallbacksCall,
  setCallbacksCall,
  modelSettingsCall,
  adminGlobalActivityExceptions,
  adminGlobalActivityExceptionsPerDeployment,
  allEndUsersCall,
} from "@/components/networking";
import { Form } from "antd";
import { Typography } from "antd";
import { RefreshIcon } from "@heroicons/react/outline";
import type { UploadProps } from "antd";
import { Team } from "@/components/key_team_helpers/key_list";
import TeamInfoView from "../../../components/team/team_info";
import { Providers, getPlaceholder, getProviderModels } from "@/components/provider_info_helpers";
import ModelInfoView from "../../../components/model_info_view";
import AddModelTab from "../../../components/add_model/add_model_tab";

import HealthCheckComponent from "../../../components/model_dashboard/HealthCheckComponent";
import PassThroughSettings from "../../../components/pass_through_settings";
import ModelGroupAliasSettings from "../../../components/model_group_alias_settings";
import { all_admin_roles } from "@/utils/roles";
import NotificationsManager from "../../../components/molecules/notifications_manager";
import AllModelsTab from "@/app/(dashboard)/models-and-endpoints/components/AllModelsTab";
import PriceDataManagementTab from "@/app/(dashboard)/models-and-endpoints/components/PriceDataManagementTab";
import ModelRetrySettingsTab from "@/app/(dashboard)/models-and-endpoints/components/ModelRetrySettingsTab";
import ModelAnalyticsTab from "@/app/(dashboard)/models-and-endpoints/components/ModelAnalyticsTab/ModelAnalyticsTab";

interface ModelDashboardProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
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
  accessToken,
  token,
  userRole,
  userID,
  modelData = { data: [] },
  keys,
  setModelData,
  premiumUser,
  teams,
}) => {
  const [addModelForm] = Form.useForm();
  const [modelMap, setModelMap] = useState<any>(null);
  const [lastRefreshed, setLastRefreshed] = useState("");

  const [providerModels, setProviderModels] = useState<Array<string>>([]); // Explicitly typing providerModels as a string array

  const [providerSettings, setProviderSettings] = useState<ProviderSettings[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.OpenAI);
  const [editModalVisible, setEditModalVisible] = useState<boolean>(false);

  const [selectedModel, setSelectedModel] = useState<any>(null);
  const [availableModelGroups, setAvailableModelGroups] = useState<Array<string>>([]);
  const [availableModelAccessGroups, setAvailableModelAccessGroups] = useState<Array<string>>([]);
  const [selectedModelGroup, setSelectedModelGroup] = useState<string | null>(null);
  const [modelMetrics, setModelMetrics] = useState<any[]>([]);
  const [modelMetricsCategories, setModelMetricsCategories] = useState<any[]>([]);
  const [streamingModelMetrics, setStreamingModelMetrics] = useState<any[]>([]);
  const [streamingModelMetricsCategories, setStreamingModelMetricsCategories] = useState<any[]>([]);
  const [modelExceptions, setModelExceptions] = useState<any[]>([]);
  const [allExceptions, setAllExceptions] = useState<any[]>([]);
  const [slowResponsesData, setSlowResponsesData] = useState<any[]>([]);
  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });

  const [modelGroupRetryPolicy, setModelGroupRetryPolicy] = useState<RetryPolicyObject | null>(null);
  const [globalRetryPolicy, setGlobalRetryPolicy] = useState<GlobalRetryPolicyObject | null>(null);
  const [defaultRetry, setDefaultRetry] = useState<number>(0);

  const [globalExceptionData, setGlobalExceptionData] = useState<GlobalExceptionActivityData>(
    {} as GlobalExceptionActivityData,
  );
  const [globalExceptionPerDeployment, setGlobalExceptionPerDeployment] = useState<any[]>([]);

  const [showAdvancedFilters, setShowAdvancedFilters] = useState<boolean>(false);
  const [selectedAPIKey, setSelectedAPIKey] = useState<any | null>(null);
  const [selectedCustomer, setSelectedCustomer] = useState<any | null>(null);

  const [allEndUsers, setAllEndUsers] = useState<any[]>([]);

  const [credentialsList, setCredentialsList] = useState<CredentialItem[]>([]);

  // Model Group Alias state
  const [modelGroupAlias, setModelGroupAlias] = useState<{ [key: string]: string }>({});

  // Add state for advanced settings visibility
  const [showAdvancedSettings, setShowAdvancedSettings] = useState<boolean>(false);

  // Add these state variables
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [editModel, setEditModel] = useState<boolean>(false);

  const [selectedTeamId, setSelectedTeamId] = useState<string | null>(null);
  const [selectedTeam, setSelectedTeam] = useState<string | null>(null);

  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const [selectedTabIndex, setSelectedTabIndex] = useState(0);
  const setProviderModelsFn = (provider: Providers) => {
    const _providerModels = getProviderModels(provider, modelMap);
    setProviderModels(_providerModels);
  };

  const fetchCredentials = async (accessToken: string) => {
    try {
      const response: CredentialsResponse = await credentialListCall(accessToken);
      setCredentialsList(response.credentials);
    } catch (error) {
      NotificationsManager.fromBackend("Error fetching credentials");
    }
  };

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  const uploadProps: UploadProps = {
    name: "file",
    accept: ".json",
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
      // Prevent upload
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
    // Update the 'lastRefreshed' state to the current date and time
    const currentDate = new Date();
    setLastRefreshed(currentDate.toLocaleString());
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
        // Only update global retry policy
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
    if (!accessToken || !token || !userRole || !userID) {
      return;
    }
    const fetchData = async () => {
      try {
        // Replace with your actual API call for model data
        const modelDataResponse = await modelInfoCall(accessToken, userID, userRole);
        setModelData(modelDataResponse);
        const _providerSettings = await modelSettingsCall(accessToken);
        if (_providerSettings) {
          setProviderSettings(_providerSettings);
        }

        // loop through modelDataResponse and get all`model_name` values
        let all_model_groups: Set<string> = new Set();
        for (let i = 0; i < modelDataResponse.data.length; i++) {
          const model = modelDataResponse.data[i];
          all_model_groups.add(model.model_name);
        }
        let _array_model_groups = Array.from(all_model_groups);
        // sort _array_model_groups alphabetically
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

        let _initial_model_group = "all";
        if (_array_model_groups.length > 0) {
          _initial_model_group = _array_model_groups[_array_model_groups.length - 1];
        }

        const modelMetricsResponse = await modelMetricsCall(
          accessToken,
          userID,
          userRole,
          _initial_model_group,
          dateValue.from?.toISOString(),
          dateValue.to?.toISOString(),
          selectedAPIKey?.token,
          selectedCustomer,
        );

        setModelMetrics(modelMetricsResponse.data);
        setModelMetricsCategories(modelMetricsResponse.all_api_bases);

        const streamingModelMetricsResponse = await streamingModelMetricsCall(
          accessToken,
          _initial_model_group,
          dateValue.from?.toISOString(),
          dateValue.to?.toISOString(),
        );

        // Assuming modelMetricsResponse now contains the metric data for the specified model group
        setStreamingModelMetrics(streamingModelMetricsResponse.data);
        setStreamingModelMetricsCategories(streamingModelMetricsResponse.all_api_bases);

        const modelExceptionsResponse = await modelExceptionsCall(
          accessToken,
          userID,
          userRole,
          _initial_model_group,
          dateValue.from?.toISOString(),
          dateValue.to?.toISOString(),
          selectedAPIKey?.token,
          selectedCustomer,
        );
        setModelExceptions(modelExceptionsResponse.data);
        setAllExceptions(modelExceptionsResponse.exception_types);

        const slowResponses = await modelMetricsSlowResponsesCall(
          accessToken,
          userID,
          userRole,
          _initial_model_group,
          dateValue.from?.toISOString(),
          dateValue.to?.toISOString(),
          selectedAPIKey?.token,
          selectedCustomer,
        );

        const dailyExceptions = await adminGlobalActivityExceptions(
          accessToken,
          dateValue.from?.toISOString().split("T")[0],
          dateValue.to?.toISOString().split("T")[0],
          _initial_model_group,
        );

        setGlobalExceptionData(dailyExceptions);

        const dailyExceptionsPerDeplyment = await adminGlobalActivityExceptionsPerDeployment(
          accessToken,
          dateValue.from?.toISOString().split("T")[0],
          dateValue.to?.toISOString().split("T")[0],
          _initial_model_group,
        );

        setGlobalExceptionPerDeployment(dailyExceptionsPerDeplyment);
        setSlowResponsesData(slowResponses);
        let all_end_users_data = await allEndUsersCall(accessToken);
        setAllEndUsers(all_end_users_data?.map((u: any) => u.user_id));
        const routerSettingsInfo = await getCallbacksCall(accessToken, userID, userRole);
        let router_settings = routerSettingsInfo.router_settings;

        let model_group_retry_policy = router_settings.model_group_retry_policy;
        let default_retries = router_settings.num_retries;

        setModelGroupRetryPolicy(model_group_retry_policy);
        setGlobalRetryPolicy(router_settings.retry_policy);
        setDefaultRetry(default_retries);

        // Set model group alias
        const model_group_alias = router_settings.model_group_alias || {};
        setModelGroupAlias(model_group_alias);
      } catch (error) {
        NotificationsManager.fromBackend("Error fetching model data: " + error);
      }
    };

    if (accessToken && token && userRole && userID) {
      fetchData();
    }

    const fetchModelMap = async () => {
      const data = await modelCostMap(accessToken);
      setModelMap(data);
    };
    if (modelMap == null) {
      fetchModelMap();
    }

    handleRefreshClick();
  }, [accessToken, token, userRole, userID, modelMap, lastRefreshed, selectedTeam]);

  if (!modelData) {
    return <div>Loading...</div>;
  }

  if (!accessToken || !token || !userRole || !userID) {
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

    let defaultProvider = "openai";
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
      if (modelMap !== null && modelMap !== undefined) {
        if (typeof modelMap == "object" && model in modelMap) {
          return modelMap[model]["litellm_provider"];
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
  const customTooltip = (props: any) => {
    const { payload, active } = props;
    if (!active || !payload) return null;

    // Extract the date from the first item in the payload array
    const date = payload[0]?.payload?.date;

    // Sort the payload array by category.value in descending order
    let sortedPayload = payload.sort((a: any, b: any) => b.value - a.value);

    // Only show the top 5, the 6th one should be called "X other categories" depending on how many categories were not shown
    if (sortedPayload.length > 5) {
      let remainingItems = sortedPayload.length - 5;
      sortedPayload = sortedPayload.slice(0, 5);
      sortedPayload.push({
        dataKey: `${remainingItems} other deployments`,
        value: payload.slice(5).reduce((acc: number, curr: any) => acc + curr.value, 0),
        color: "gray",
      });
    }

    return (
      <div className="w-150 rounded-tremor-default border border-tremor-border bg-tremor-background p-2 text-tremor-default shadow-tremor-dropdown">
        {date && <p className="text-tremor-content-emphasis mb-2">Date: {date}</p>}
        {sortedPayload.map((category: any, idx: number) => {
          const roundedValue = parseFloat(category.value.toFixed(5));
          const displayValue = roundedValue === 0 && category.value > 0 ? "<0.00001" : roundedValue.toFixed(5);
          return (
            <div key={idx} className="flex justify-between">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 mt-1 rounded-full bg-${category.color}-500`} />
                <p className="text-tremor-content">{category.dataKey}</p>
              </div>
              <p className="font-medium text-tremor-content-emphasis text-righ ml-2">{displayValue}</p>
            </div>
          );
        })}
      </div>
    );
  };

  const handleOk = () => {
    addModelForm
      .validateFields()
      .then((values: any) => {
        handleAddModelSubmit(values, accessToken, addModelForm, handleRefreshClick);
      })
      .catch((error: any) => {
        const errorMessages =
          error.errorFields
            ?.map((field: any) => {
              return `${field.name.join(".")}: ${field.errors.join(", ")}`;
            })
            .join(" | ") || "Unknown validation error";
        NotificationsManager.fromBackend(`Please fill in the following required fields: ${errorMessages}`);
      });
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
              editModel={true}
              onClose={() => {
                setSelectedModelId(null);
                setEditModel(false);
              }}
              modelData={modelData.data.find((model: any) => model.model_info.id === selectedModelId)}
              accessToken={accessToken}
              userID={userID}
              userRole={userRole}
              setEditModalVisible={setEditModalVisible}
              setSelectedModel={setSelectedModel}
              onModelUpdate={(updatedModel) => {
                // Update the model in the modelData.data array
                const updatedModelData = {
                  ...modelData,
                  data: modelData.data.map((model: any) =>
                    model.model_info.id === updatedModel.model_info.id ? updatedModel : model,
                  ),
                };
                setModelData(updatedModelData);
                // Trigger a refresh to update UI
                handleRefreshClick();
              }}
              modelAccessGroups={availableModelAccessGroups}
            />
          ) : (
            <TabGroup index={selectedTabIndex} onIndexChange={setSelectedTabIndex} className="gap-2 h-[75vh] w-full ">
              <TabList className="flex justify-between mt-2 w-full items-center">
                <div className="flex">
                  {all_admin_roles.includes(userRole) ? <Tab>All Models</Tab> : <Tab>Your Models</Tab>}
                  <Tab>Add Model</Tab>
                  {all_admin_roles.includes(userRole) && <Tab>LLM Credentials</Tab>}
                  {all_admin_roles.includes(userRole) && <Tab>Pass-Through Endpoints</Tab>}
                  {all_admin_roles.includes(userRole) && <Tab>Health Status</Tab>}
                  {all_admin_roles.includes(userRole) && <Tab>Model Analytics</Tab>}
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
                  setEditModel={setEditModel}
                  modelData={modelData}
                />
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
                    premiumUser={premiumUser}
                  />
                </TabPanel>
                <TabPanel>
                  <CredentialsPanel
                    accessToken={accessToken}
                    uploadProps={uploadProps}
                    credentialList={credentialsList}
                    fetchCredentials={fetchCredentials}
                  />
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
                <ModelAnalyticsTab
                  dateValue={dateValue}
                  setDateValue={setDateValue}
                  selectedModelGroup={selectedModelGroup}
                  availableModelGroups={availableModelGroups}
                  setShowAdvancedFilters={setShowAdvancedFilters}
                  modelMetrics={modelMetrics}
                  modelMetricsCategories={modelMetricsCategories}
                  streamingModelMetrics={streamingModelMetrics}
                  streamingModelMetricsCategories={streamingModelMetricsCategories}
                  customTooltip={customTooltip}
                  slowResponsesData={slowResponsesData}
                  modelExceptions={modelExceptions}
                  globalExceptionData={globalExceptionData}
                  allExceptions={allExceptions}
                  globalExceptionPerDeployment={globalExceptionPerDeployment}
                  allEndUsers={allEndUsers}
                  keys={keys}
                  setSelectedAPIKey={setSelectedAPIKey}
                  setSelectedCustomer={setSelectedCustomer}
                  teams={teams}
                  selectedAPIKey={selectedAPIKey}
                  selectedCustomer={selectedCustomer}
                  selectedTeam={selectedTeam}
                  setAllExceptions={setAllExceptions}
                  setGlobalExceptionData={setGlobalExceptionData}
                  setGlobalExceptionPerDeployment={setGlobalExceptionPerDeployment}
                  setModelExceptions={setModelExceptions}
                  setModelMetrics={setModelMetrics}
                  setModelMetricsCategories={setModelMetricsCategories}
                  setSelectedModelGroup={setSelectedModelGroup}
                  setSlowResponsesData={setSlowResponsesData}
                  setStreamingModelMetrics={setStreamingModelMetrics}
                  setStreamingModelMetricsCategories={setStreamingModelMetricsCategories}
                />
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
                <PriceDataManagementTab setModelMap={setModelMap} />
              </TabPanels>
            </TabGroup>
          )}
        </Col>
      </Grid>
    </div>
  );
};

export default ModelsAndEndpointsView;
