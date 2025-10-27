import React, { useState, useEffect, useRef, useMemo } from "react";
import {
  Card,
  Title,
  Subtitle,
  Table,
  TableHead,
  TableRow,
  TableHeaderCell,
  TableCell,
  TableBody,
  Text,
  Grid,
  Col,
} from "@tremor/react";
import { CredentialItem, credentialListCall, CredentialsResponse } from "../networking";

import { handleAddModelSubmit } from "../add_model/handle_add_model_submit";

import CredentialsPanel from "@/components/model_add/credentials";
import { getDisplayModelName } from "../view_model/model_name_display";
import { TabPanel, TabPanels, TabGroup, TabList, Tab, Icon } from "@tremor/react";
import { Select, SelectItem, DateRangePickerValue } from "@tremor/react";
import UsageDatePicker from "../shared/usage_date_picker";
import {
  modelInfoCall,
  modelCostMap,
  healthCheckCall,
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
} from "../networking";
import { BarChart, AreaChart } from "@tremor/react";
import { Popover, Form, InputNumber } from "antd";
import { Button } from "@tremor/react";
import { Typography } from "antd";
import { RefreshIcon, FilterIcon } from "@heroicons/react/outline";
import { InfoCircleOutlined } from "@ant-design/icons";
import type { UploadProps } from "antd";
import TimeToFirstToken from "../model_metrics/time_to_first_token";
import { Team } from "../key_team_helpers/key_list";
import TeamInfoView from "../team/team_info";
import { Providers, provider_map, getPlaceholder, getProviderModels } from "../provider_info_helpers";
import ModelInfoView from "../model_info_view";
import AddModelTab from "../add_model/add_model_tab";

import { ModelDataTable } from "../model_dashboard/table";
import { columns } from "../molecules/models/columns";
import PriceDataReload from "../price_data_reload";
import HealthCheckComponent from "../model_dashboard/HealthCheckComponent";
import PassThroughSettings from "../pass_through_settings";
import ModelGroupAliasSettings from "../model_group_alias_settings";
import { all_admin_roles } from "@/utils/roles";
import { Table as TableInstance, PaginationState } from "@tanstack/react-table";
import NotificationsManager from "../molecules/notifications_manager";

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

const retry_policy_map: Record<string, string> = {
  "BadRequestError (400)": "BadRequestErrorRetries",
  "AuthenticationError  (401)": "AuthenticationErrorRetries",
  "TimeoutError (408)": "TimeoutErrorRetries",
  "RateLimitError (429)": "RateLimitErrorRetries",
  "ContentPolicyViolationError (400)": "ContentPolicyViolationErrorRetries",
  "InternalServerError (500)": "InternalServerErrorRetries",
};

const OldModelDashboard: React.FC<ModelDashboardProps> = ({
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
  const [autoRouterForm] = Form.useForm();
  const [modelMap, setModelMap] = useState<any>(null);
  const [lastRefreshed, setLastRefreshed] = useState("");

  const [providerModels, setProviderModels] = useState<Array<string>>([]); // Explicitly typing providerModels as a string array

  const [providerSettings, setProviderSettings] = useState<ProviderSettings[]>([]);
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.OpenAI);
  const [healthCheckResponse, setHealthCheckResponse] = useState<any>(null);
  const [isHealthCheckLoading, setIsHealthCheckLoading] = useState<boolean>(false);
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

  const [selectedTeamFilter, setSelectedTeamFilter] = useState<string | null>(null);
  const [selectedModelAccessGroupFilter, setSelectedModelAccessGroupFilter] = useState<string | null>(null);

  const [modelNameSearch, setModelNameSearch] = useState<string>("");

  // Add new state for current team and model view mode
  const [currentTeam, setCurrentTeam] = useState<string>("personal"); // 'personal' or team_id
  const [modelViewMode, setModelViewMode] = useState<"current_team" | "all">("current_team");

  // Add state for showing/hiding filters
  const [showFilters, setShowFilters] = useState<boolean>(false);

  const [showColumnDropdown, setShowColumnDropdown] = useState(false);

  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [expandedRows, setExpandedRows] = useState<Set<string>>(new Set());
  const dropdownRef = useRef<HTMLDivElement>(null);
  const tableRef = useRef<TableInstance<any>>(null);

  // Pagination state
  const [pagination, setPagination] = useState<PaginationState>({
    pageIndex: 0,
    pageSize: 50,
  });
  const [selectedTabIndex, setSelectedTabIndex] = useState(0);

  const handleCreateNewModelClick = () => {
    if (selectedModelId) {
      setSelectedModelId(null);
    }
    setSelectedTabIndex(1);
  };

  const resetFilters = () => {
    setModelNameSearch("");
    setSelectedModelGroup("all");
    setSelectedModelAccessGroupFilter(null);
    setCurrentTeam("personal");
    setModelViewMode("current_team");
    setPagination({ pageIndex: 0, pageSize: 50 });
  };

  // Memoize filtered data to prevent unnecessary re-calculations
  const filteredData = useMemo(() => {
    if (!modelData || !modelData.data || modelData.data.length === 0) {
      return [];
    }

    return modelData.data.filter((model: any) => {
      const searchMatch =
        modelNameSearch === "" || model.model_name.toLowerCase().includes(modelNameSearch.toLowerCase());

      const modelNameMatch =
        selectedModelGroup === "all" ||
        model.model_name === selectedModelGroup ||
        !selectedModelGroup ||
        (selectedModelGroup === "wildcard" && model.model_name?.includes("*"));

      const accessGroupMatch =
        selectedModelAccessGroupFilter === "all" ||
        model.model_info["access_groups"]?.includes(selectedModelAccessGroupFilter) ||
        !selectedModelAccessGroupFilter;

      let teamAccessMatch = true;
      if (modelViewMode === "current_team") {
        if (currentTeam === "personal") {
          teamAccessMatch = model.model_info?.direct_access === true;
        } else {
          teamAccessMatch = model.model_info?.access_via_team_ids?.includes(currentTeam) === true;
        }
      }

      return searchMatch && modelNameMatch && accessGroupMatch && teamAccessMatch;
    });
  }, [modelData, modelNameSearch, selectedModelGroup, selectedModelAccessGroupFilter, currentTeam, modelViewMode]);

  // Memoize paginated data
  const paginatedData = useMemo(() => {
    const startIndex = pagination.pageIndex * pagination.pageSize;
    const endIndex = startIndex + pagination.pageSize;
    return filteredData.slice(startIndex, endIndex);
  }, [filteredData, pagination.pageIndex, pagination.pageSize]);

  // Reset pagination when filters change
  useEffect(() => {
    setPagination((prev) => ({ ...prev, pageIndex: 0 }));
  }, [modelNameSearch, selectedModelGroup, selectedModelAccessGroupFilter, currentTeam, modelViewMode]);

  const setProviderModelsFn = (provider: Providers) => {
    const _providerModels = getProviderModels(provider, modelMap);
    setProviderModels(_providerModels);
    console.log(`providerModels: ${_providerModels}`);
  };

  const updateModelMetrics = async (
    modelGroup: string | null,
    startTime: Date | undefined,
    endTime: Date | undefined,
  ) => {
    console.log("Updating model metrics for group:", modelGroup);
    if (!accessToken || !userID || !userRole || !startTime || !endTime) {
      return;
    }
    console.log("inside updateModelMetrics - startTime:", startTime, "endTime:", endTime);
    setSelectedModelGroup(modelGroup);

    let selected_token = selectedAPIKey?.token;
    if (selected_token === undefined) {
      selected_token = null;
    }

    let selected_customer = selectedCustomer;
    if (selected_customer === undefined) {
      selected_customer = null;
    }

    try {
      const modelMetricsResponse = await modelMetricsCall(
        accessToken,
        userID,
        userRole,
        modelGroup,
        startTime.toISOString(),
        endTime.toISOString(),
        selected_token,
        selected_customer,
      );
      console.log("Model metrics response:", modelMetricsResponse);

      // Assuming modelMetricsResponse now contains the metric data for the specified model group
      setModelMetrics(modelMetricsResponse.data);
      setModelMetricsCategories(modelMetricsResponse.all_api_bases);

      const streamingModelMetricsResponse = await streamingModelMetricsCall(
        accessToken,
        modelGroup,
        startTime.toISOString(),
        endTime.toISOString(),
      );

      // Assuming modelMetricsResponse now contains the metric data for the specified model group
      setStreamingModelMetrics(streamingModelMetricsResponse.data);
      setStreamingModelMetricsCategories(streamingModelMetricsResponse.all_api_bases);

      const modelExceptionsResponse = await modelExceptionsCall(
        accessToken,
        userID,
        userRole,
        modelGroup,
        startTime.toISOString(),
        endTime.toISOString(),
        selected_token,
        selected_customer,
      );
      console.log("Model exceptions response:", modelExceptionsResponse);
      setModelExceptions(modelExceptionsResponse.data);
      setAllExceptions(modelExceptionsResponse.exception_types);

      const slowResponses = await modelMetricsSlowResponsesCall(
        accessToken,
        userID,
        userRole,
        modelGroup,
        startTime.toISOString(),
        endTime.toISOString(),
        selected_token,
        selected_customer,
      );

      console.log("slowResponses:", slowResponses);

      setSlowResponsesData(slowResponses);

      if (modelGroup) {
        const dailyExceptions = await adminGlobalActivityExceptions(
          accessToken,
          startTime?.toISOString().split("T")[0],
          endTime?.toISOString().split("T")[0],
          modelGroup,
        );

        setGlobalExceptionData(dailyExceptions);

        const dailyExceptionsPerDeplyment = await adminGlobalActivityExceptionsPerDeployment(
          accessToken,
          startTime?.toISOString().split("T")[0],
          endTime?.toISOString().split("T")[0],
          modelGroup,
        );

        setGlobalExceptionPerDeployment(dailyExceptionsPerDeplyment);
      }
    } catch (error) {
      console.error("Failed to fetch model metrics", error);
    }
  };

  const fetchCredentials = async (accessToken: string) => {
    try {
      const response: CredentialsResponse = await credentialListCall(accessToken);
      console.log(`credentials: ${JSON.stringify(response)}`);
      setCredentialsList(response.credentials);
    } catch (error) {
      console.error("Error fetching credentials:", error);
    }
  };

  useEffect(() => {
    updateModelMetrics(selectedModelGroup, dateValue.from, dateValue.to);
  }, [selectedAPIKey, selectedCustomer, selectedTeam]);

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsDropdownOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  function formatCreatedAt(createdAt: string | null) {
    if (createdAt) {
      const date = new Date(createdAt);
      const options = { month: "long", day: "numeric", year: "numeric" };
      return date.toLocaleDateString("en-US");
    }
    return null;
  }

  const handleEditClick = (model: any) => {
    setSelectedModel(model);
    setEditModalVisible(true);
  };

  const handleEditCancel = () => {
    setEditModalVisible(false);
    setSelectedModel(null);
  };

  const uploadProps: UploadProps = {
    name: "file",
    accept: ".json",
    beforeUpload: (file) => {
      if (file.type === "application/json") {
        const reader = new FileReader();
        reader.onload = (e) => {
          if (e.target) {
            const jsonStr = e.target.result as string;
            console.log(`Resetting vertex_credentials to JSON; jsonStr: ${jsonStr}`);
            addModelForm.setFieldsValue({ vertex_credentials: jsonStr });
            console.log("Form values right after setting:", addModelForm.getFieldsValue());
          }
        };
        reader.readAsText(file);
      }
      // Prevent upload
      return false;
    },
    onChange(info) {
      console.log("Upload onChange triggered with values:", info);
      console.log("Current form values:", addModelForm.getFieldsValue());

      if (info.file.status !== "uploading") {
        console.log(info.file, info.fileList);
      }
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
      console.error("Access token is missing");
      return;
    }

    try {
      const payload: any = {
        router_settings: {},
      };

      if (selectedModelGroup === "global") {
        // Only update global retry policy
        console.log("Saving global retry policy:", globalRetryPolicy);
        if (globalRetryPolicy) {
          payload.router_settings.retry_policy = globalRetryPolicy;
        }
        NotificationsManager.success("Global retry settings saved successfully");
      } else {
        // Only update model group retry policy
        console.log("Saving model group retry policy for", selectedModelGroup, ":", modelGroupRetryPolicy);
        if (modelGroupRetryPolicy) {
          payload.router_settings.model_group_retry_policy = modelGroupRetryPolicy;
        }
        NotificationsManager.success(`Retry settings saved successfully for ${selectedModelGroup}`);
      }

      await setCallbacksCall(accessToken, payload);
    } catch (error) {
      console.error("Failed to save retry settings:", error);
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
        console.log("Model data response:", modelDataResponse.data);
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
        console.log("all_model_groups:", all_model_groups);
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

        console.log("array_model_groups:", _array_model_groups);
        let _initial_model_group = "all";
        if (_array_model_groups.length > 0) {
          // set selectedModelGroup to the last model group
          _initial_model_group = _array_model_groups[_array_model_groups.length - 1];
          console.log("_initial_model_group:", _initial_model_group);
          //setSelectedModelGroup(_initial_model_group);
        }

        console.log("selectedModelGroup:", selectedModelGroup);

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

        console.log("Model metrics response:", modelMetricsResponse);
        // Sort by latency (avg_latency_per_token)

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
        console.log("Model exceptions response:", modelExceptionsResponse);
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

        console.log("dailyExceptions:", dailyExceptions);

        console.log("dailyExceptionsPerDeplyment:", dailyExceptionsPerDeplyment);

        console.log("slowResponses:", slowResponses);

        setSlowResponsesData(slowResponses);

        let all_end_users_data = await allEndUsersCall(accessToken);

        setAllEndUsers(all_end_users_data?.map((u: any) => u.user_id));

        const routerSettingsInfo = await getCallbacksCall(accessToken, userID, userRole);

        let router_settings = routerSettingsInfo.router_settings;

        console.log("routerSettingsInfo:", router_settings);

        let model_group_retry_policy = router_settings.model_group_retry_policy;
        let default_retries = router_settings.num_retries;

        console.log("model_group_retry_policy:", model_group_retry_policy);
        console.log("default_retries:", default_retries);
        setModelGroupRetryPolicy(model_group_retry_policy);
        setGlobalRetryPolicy(router_settings.retry_policy);
        setDefaultRetry(default_retries);

        // Set model group alias
        const model_group_alias = router_settings.model_group_alias || {};
        setModelGroupAlias(model_group_alias);
      } catch (error) {
        console.error("There was an error fetching the model data", error);
      }
    };

    if (accessToken && token && userRole && userID) {
      fetchData();
    }

    const fetchModelMap = async () => {
      const data = await modelCostMap(accessToken);
      console.log(`received model cost map data: ${Object.keys(data)}`);
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
      console.log(`GET PROVIDER CALLED! - ${modelMap}`);
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

    console.log(modelData.data[i]);
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

  const runHealthCheck = async () => {
    try {
      NotificationsManager.info("Running health check...");
      setIsHealthCheckLoading(true);
      setHealthCheckResponse(null);
      const response = await healthCheckCall(accessToken);
      setHealthCheckResponse(response);
    } catch (error) {
      console.error("Error running health check:", error);
      setHealthCheckResponse("Error running health check");
    } finally {
      setIsHealthCheckLoading(false);
    }
  };

  const FilterByContent = (
    <div>
      <Text className="mb-1">Select API Key Name</Text>

      {premiumUser ? (
        <div>
          <Select defaultValue="all-keys">
            <SelectItem
              key="all-keys"
              value="all-keys"
              onClick={() => {
                setSelectedAPIKey(null);
              }}
            >
              All Keys
            </SelectItem>
            {keys?.map((key: any, index: number) => {
              if (key && key["key_alias"] !== null && key["key_alias"].length > 0) {
                return (
                  <SelectItem
                    key={index}
                    value={String(index)}
                    onClick={() => {
                      setSelectedAPIKey(key);
                    }}
                  >
                    {key["key_alias"]}
                  </SelectItem>
                );
              }
              return null;
            })}
          </Select>

          <Text className="mt-1">Select Customer Name</Text>

          <Select defaultValue="all-customers">
            <SelectItem
              key="all-customers"
              value="all-customers"
              onClick={() => {
                setSelectedCustomer(null);
              }}
            >
              All Customers
            </SelectItem>
            {allEndUsers?.map((user: any, index: number) => {
              return (
                <SelectItem
                  key={index}
                  value={user}
                  onClick={() => {
                    setSelectedCustomer(user);
                  }}
                >
                  {user}
                </SelectItem>
              );
            })}
          </Select>

          <Text className="mt-1">Select Team</Text>

          <Select
            className="w-64 relative z-50"
            defaultValue="all"
            value={selectedTeamFilter ?? "all"}
            onValueChange={(value) => setSelectedTeamFilter(value === "all" ? null : value)}
          >
            <SelectItem value="all">All Teams</SelectItem>
            {teams
              ?.filter((team) => team.team_id)
              .map((team) => (
                <SelectItem key={team.team_id} value={team.team_id}>
                  {team.team_alias
                    ? `${team.team_alias} (${team.team_id.slice(0, 8)}...)`
                    : `Team ${team.team_id.slice(0, 8)}...`}
                </SelectItem>
              ))}
          </Select>
        </div>
      ) : (
        <div>
          {/* ... existing non-premium user content ... */}
          <Text className="mt-1">Select Team</Text>

          <Select
            className="w-64 relative z-50"
            defaultValue="all"
            value={selectedTeamFilter ?? "all"}
            onValueChange={(value) => setSelectedTeamFilter(value === "all" ? null : value)}
          >
            <SelectItem value="all">All Teams</SelectItem>
            {teams
              ?.filter((team) => team.team_id)
              .map((team) => (
                <SelectItem key={team.team_id} value={team.team_id}>
                  {team.team_alias
                    ? `${team.team_alias} (${team.team_id.slice(0, 8)}...)`
                    : `Team ${team.team_id.slice(0, 8)}...`}
                </SelectItem>
              ))}
          </Select>
        </div>
      )}
    </div>
  );

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
    console.log("🚀 handleOk called from model dashboard!");
    console.log("Current form values:", addModelForm.getFieldsValue());

    addModelForm
      .validateFields()
      .then((values: any) => {
        console.log("✅ Validation passed, submitting:", values);
        handleAddModelSubmit(values, accessToken, addModelForm, handleRefreshClick);
      })
      .catch((error: any) => {
        console.error("❌ Validation failed:", error);
        console.error("Form errors:", error.errorFields);
        const errorMessages =
          error.errorFields
            ?.map((field: any) => {
              return `${field.name.join(".")}: ${field.errors.join(", ")}`;
            })
            .join(" | ") || "Unknown validation error";
        NotificationsManager.fromBackend(`Please fill in the following required fields: ${errorMessages}`);
      });
  };

  console.log(`selectedProvider: ${selectedProvider}`);
  console.log(`providerModels.length: ${providerModels.length}`);

  const providerKey = Object.keys(Providers).find(
    (key) => (Providers as { [index: string]: any })[key] === selectedProvider,
  );

  let dynamicProviderForm: ProviderSettings | undefined = undefined;
  if (providerKey && providerSettings) {
    dynamicProviderForm = providerSettings.find((provider) => provider.name === provider_map[providerKey]);
  }

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
                <TabPanel>
                  <Grid>
                    <div className="flex flex-col space-y-4">
                      <div className="bg-white rounded-lg shadow">
                        {/* Current Team and View Mode Selector - Prominent Section */}
                        <div className="border-b px-6 py-4 bg-gray-50">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-4">
                              <Text className="text-lg font-semibold text-gray-900">Current Team:</Text>
                              <Select
                                className="w-80"
                                defaultValue="personal"
                                value={currentTeam}
                                onValueChange={(value) => setCurrentTeam(value)}
                              >
                                <SelectItem value="personal">
                                  <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 bg-blue-500 rounded-full"></div>
                                    <span className="font-medium">Personal</span>
                                  </div>
                                </SelectItem>
                                {teams
                                  ?.filter((team) => team.team_id)
                                  .map((team) => (
                                    <SelectItem key={team.team_id} value={team.team_id}>
                                      <div className="flex items-center gap-2">
                                        <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                                        <span className="font-medium">
                                          {team.team_alias
                                            ? `${team.team_alias.slice(0, 30)}...`
                                            : `Team ${team.team_id.slice(0, 30)}...`}
                                        </span>
                                      </div>
                                    </SelectItem>
                                  ))}
                              </Select>
                            </div>

                            <div className="flex items-center gap-4">
                              <Text className="text-lg font-semibold text-gray-900">View:</Text>
                              <Select
                                className="w-64"
                                defaultValue="current_team"
                                value={modelViewMode}
                                onValueChange={(value) => setModelViewMode(value as "current_team" | "all")}
                              >
                                <SelectItem value="current_team">
                                  <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 bg-purple-500 rounded-full"></div>
                                    <span className="font-medium">Current Team Models</span>
                                  </div>
                                </SelectItem>
                                <SelectItem value="all">
                                  <div className="flex items-center gap-2">
                                    <div className="w-2 h-2 bg-gray-500 rounded-full"></div>
                                    <span className="font-medium">All Available Models</span>
                                  </div>
                                </SelectItem>
                              </Select>
                            </div>
                          </div>

                          {modelViewMode === "current_team" && (
                            <div className="flex items-start gap-2 mt-3">
                              <InfoCircleOutlined className="text-gray-400 mt-0.5 flex-shrink-0 text-xs" />
                              <div className="text-xs text-gray-500">
                                {currentTeam === "personal" ? (
                                  <span>
                                    To access these models: Create a Virtual Key without selecting a team on the{" "}
                                    <a
                                      href="/?login=success&page=api-keys"
                                      className="text-gray-600 hover:text-gray-800 underline"
                                    >
                                      Virtual Keys page
                                    </a>
                                  </span>
                                ) : (
                                  <span>
                                    To access these models: Create a Virtual Key and select Team as &quot;
                                    {currentTeam}&quot; on the{" "}
                                    <a
                                      href="/?login=success&page=api-keys"
                                      className="text-gray-600 hover:text-gray-800 underline"
                                    >
                                      Virtual Keys page
                                    </a>
                                  </span>
                                )}
                              </div>
                            </div>
                          )}
                        </div>

                        {/* Search and Filter Controls */}
                        <div className="border-b px-6 py-4">
                          <div className="flex flex-col space-y-4">
                            {/* Search and Filter Controls */}
                            <div className="flex flex-wrap items-center gap-3">
                              {/* Model Name Search */}
                              <div className="relative w-64">
                                <input
                                  type="text"
                                  placeholder="Search model names..."
                                  className="w-full px-3 py-2 pl-8 border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                                  value={modelNameSearch}
                                  onChange={(e) => setModelNameSearch(e.target.value)}
                                />
                                <svg
                                  className="absolute left-2.5 top-2.5 h-4 w-4 text-gray-500"
                                  fill="none"
                                  stroke="currentColor"
                                  viewBox="0 0 24 24"
                                >
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
                                  />
                                </svg>
                              </div>

                              {/* Filter Button */}
                              <button
                                className={`px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2 ${showFilters ? "bg-gray-100" : ""}`}
                                onClick={() => setShowFilters(!showFilters)}
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z"
                                  />
                                </svg>
                                Filters
                              </button>

                              {/* Reset Filters Button */}
                              <button
                                className="px-3 py-2 text-sm border rounded-md hover:bg-gray-50 flex items-center gap-2"
                                onClick={resetFilters}
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                                  />
                                </svg>
                                Reset Filters
                              </button>
                            </div>

                            {/* Additional Filters */}
                            {showFilters && (
                              <div className="flex flex-wrap items-center gap-3 mt-3">
                                {/* Model Name Filter */}
                                <div className="w-64">
                                  <Select
                                    value={selectedModelGroup ?? "all"}
                                    onValueChange={(value) => setSelectedModelGroup(value === "all" ? "all" : value)}
                                    placeholder="Filter by Public Model Name"
                                  >
                                    <SelectItem value="all">All Models</SelectItem>
                                    <SelectItem value="wildcard">Wildcard Models (*)</SelectItem>
                                    {availableModelGroups.map((group, idx) => (
                                      <SelectItem key={idx} value={group}>
                                        {group}
                                      </SelectItem>
                                    ))}
                                  </Select>
                                </div>

                                {/* Model Access Group Filter */}
                                <div className="w-64">
                                  <Select
                                    value={selectedModelAccessGroupFilter ?? "all"}
                                    onValueChange={(value) =>
                                      setSelectedModelAccessGroupFilter(value === "all" ? null : value)
                                    }
                                    placeholder="Filter by Model Access Group"
                                  >
                                    <SelectItem value="all">All Model Access Groups</SelectItem>
                                    {availableModelAccessGroups.map((accessGroup, idx) => (
                                      <SelectItem key={idx} value={accessGroup}>
                                        {accessGroup}
                                      </SelectItem>
                                    ))}
                                  </Select>
                                </div>
                              </div>
                            )}

                            {/* Results Count and Pagination Controls */}
                            <div className="flex justify-between items-center">
                              <span className="text-sm text-gray-700">
                                {filteredData.length > 0
                                  ? `Showing ${pagination.pageIndex * pagination.pageSize + 1} - ${Math.min(
                                      (pagination.pageIndex + 1) * pagination.pageSize,
                                      filteredData.length,
                                    )} of ${filteredData.length} results`
                                  : "Showing 0 results"}
                              </span>

                              {/* Pagination Controls */}
                              {filteredData.length > pagination.pageSize && (
                                <div className="flex items-center space-x-2">
                                  <button
                                    onClick={() =>
                                      setPagination((prev) => ({ ...prev, pageIndex: prev.pageIndex - 1 }))
                                    }
                                    disabled={pagination.pageIndex === 0}
                                    className={`px-3 py-1 text-sm border rounded-md ${
                                      pagination.pageIndex === 0
                                        ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                                        : "hover:bg-gray-50"
                                    }`}
                                  >
                                    Previous
                                  </button>

                                  <button
                                    onClick={() =>
                                      setPagination((prev) => ({ ...prev, pageIndex: prev.pageIndex + 1 }))
                                    }
                                    disabled={
                                      pagination.pageIndex >= Math.ceil(filteredData.length / pagination.pageSize) - 1
                                    }
                                    className={`px-3 py-1 text-sm border rounded-md ${
                                      pagination.pageIndex >= Math.ceil(filteredData.length / pagination.pageSize) - 1
                                        ? "bg-gray-100 text-gray-400 cursor-not-allowed"
                                        : "hover:bg-gray-50"
                                    }`}
                                  >
                                    Next
                                  </button>
                                </div>
                              )}
                            </div>
                          </div>
                        </div>

                        <ModelDataTable
                          columns={columns(
                            userRole,
                            userID,
                            premiumUser,
                            setSelectedModelId,
                            setSelectedTeamId,
                            getDisplayModelName,
                            handleEditClick,
                            handleRefreshClick,
                            setEditModel,
                            expandedRows,
                            setExpandedRows,
                          )}
                          data={paginatedData}
                          isLoading={false}
                          table={tableRef}
                        />
                      </div>
                    </div>
                  </Grid>
                </TabPanel>
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
                <TabPanel>
                  <Grid numItems={4} className="mt-2 mb-2">
                    <Col>
                      <UsageDatePicker
                        value={dateValue}
                        className="mr-2"
                        onValueChange={(value) => {
                          setDateValue(value);
                          updateModelMetrics(selectedModelGroup, value.from, value.to);
                        }}
                      />
                    </Col>
                    <Col className="ml-2">
                      <Text>Select Model Group</Text>
                      <Select
                        defaultValue={selectedModelGroup ? selectedModelGroup : availableModelGroups[0]}
                        value={selectedModelGroup ? selectedModelGroup : availableModelGroups[0]}
                      >
                        {availableModelGroups.map((group, idx) => (
                          <SelectItem
                            key={idx}
                            value={group}
                            onClick={() => updateModelMetrics(group, dateValue.from, dateValue.to)}
                          >
                            {group}
                          </SelectItem>
                        ))}
                      </Select>
                    </Col>
                    <Col>
                      <Popover
                        trigger="click"
                        content={FilterByContent}
                        overlayStyle={{
                          width: "20vw",
                        }}
                      >
                        <Button
                          icon={FilterIcon}
                          size="md"
                          variant="secondary"
                          className="mt-4 ml-2"
                          style={{
                            border: "none",
                          }}
                          onClick={() => setShowAdvancedFilters(true)}
                        ></Button>
                      </Popover>
                    </Col>
                  </Grid>

                  <Grid numItems={2}>
                    <Col>
                      <Card className="mr-2 max-h-[400px] min-h-[400px]">
                        <TabGroup>
                          <TabList variant="line" defaultValue="1">
                            <Tab value="1">Avg. Latency per Token</Tab>
                            <Tab value="2">Time to first token</Tab>
                          </TabList>
                          <TabPanels>
                            <TabPanel>
                              <p className="text-gray-500 italic"> (seconds/token)</p>
                              <Text className="text-gray-500 italic mt-1 mb-1">
                                average Latency for successfull requests divided by the total tokens
                              </Text>
                              {modelMetrics && modelMetricsCategories && (
                                <AreaChart
                                  title="Model Latency"
                                  className="h-72"
                                  data={modelMetrics}
                                  showLegend={false}
                                  index="date"
                                  categories={modelMetricsCategories}
                                  connectNulls={true}
                                  customTooltip={customTooltip}
                                />
                              )}
                            </TabPanel>
                            <TabPanel>
                              <TimeToFirstToken
                                modelMetrics={streamingModelMetrics}
                                modelMetricsCategories={streamingModelMetricsCategories}
                                customTooltip={customTooltip}
                                premiumUser={premiumUser}
                              />
                            </TabPanel>
                          </TabPanels>
                        </TabGroup>
                      </Card>
                    </Col>
                    <Col>
                      <Card className="ml-2 max-h-[400px] min-h-[400px]  overflow-y-auto">
                        <Table>
                          <TableHead>
                            <TableRow>
                              <TableHeaderCell>Deployment</TableHeaderCell>
                              <TableHeaderCell>Success Responses</TableHeaderCell>
                              <TableHeaderCell>
                                Slow Responses <p>Success Responses taking 600+s</p>
                              </TableHeaderCell>
                            </TableRow>
                          </TableHead>
                          <TableBody>
                            {slowResponsesData.map((metric, idx) => (
                              <TableRow key={idx}>
                                <TableCell>{metric.api_base}</TableCell>
                                <TableCell>{metric.total_count}</TableCell>
                                <TableCell>{metric.slow_count}</TableCell>
                              </TableRow>
                            ))}
                          </TableBody>
                        </Table>
                      </Card>
                    </Col>
                  </Grid>
                  <Grid numItems={1} className="gap-2 w-full mt-2">
                    <Card>
                      <Title>All Exceptions for {selectedModelGroup}</Title>

                      <BarChart
                        className="h-60"
                        data={modelExceptions}
                        index="model"
                        categories={allExceptions}
                        stack={true}
                        yAxisWidth={30}
                      />
                    </Card>
                  </Grid>

                  <Grid numItems={1} className="gap-2 w-full mt-2">
                    <Card>
                      <Title>All Up Rate Limit Errors (429) for {selectedModelGroup}</Title>
                      <Grid numItems={1}>
                        <Col>
                          <Subtitle
                            style={{
                              fontSize: "15px",
                              fontWeight: "normal",
                              color: "#535452",
                            }}
                          >
                            Num Rate Limit Errors {globalExceptionData.sum_num_rate_limit_exceptions}
                          </Subtitle>
                          <BarChart
                            className="h-40"
                            data={globalExceptionData.daily_data}
                            index="date"
                            colors={["rose"]}
                            categories={["num_rate_limit_exceptions"]}
                            onValueChange={(v) => console.log(v)}
                          />
                        </Col>
                        <Col></Col>
                      </Grid>
                    </Card>

                    {premiumUser ? (
                      <>
                        {globalExceptionPerDeployment.map((globalActivity, index) => (
                          <Card key={index}>
                            <Title>{globalActivity.api_base ? globalActivity.api_base : "Unknown API Base"}</Title>
                            <Grid numItems={1}>
                              <Col>
                                <Subtitle
                                  style={{
                                    fontSize: "15px",
                                    fontWeight: "normal",
                                    color: "#535452",
                                  }}
                                >
                                  Num Rate Limit Errors (429) {globalActivity.sum_num_rate_limit_exceptions}
                                </Subtitle>
                                <BarChart
                                  className="h-40"
                                  data={globalActivity.daily_data}
                                  index="date"
                                  colors={["rose"]}
                                  categories={["num_rate_limit_exceptions"]}
                                  onValueChange={(v) => console.log(v)}
                                />
                              </Col>
                            </Grid>
                          </Card>
                        ))}
                      </>
                    ) : (
                      <>
                        {globalExceptionPerDeployment &&
                          globalExceptionPerDeployment.length > 0 &&
                          globalExceptionPerDeployment.slice(0, 1).map((globalActivity, index) => (
                            <Card key={index}>
                              <Title>✨ Rate Limit Errors by Deployment</Title>
                              <p className="mb-2 text-gray-500 italic text-[12px]">
                                Upgrade to see exceptions for all deployments
                              </p>
                              <Button variant="primary" className="mb-2">
                                <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
                                  Get Free Trial
                                </a>
                              </Button>
                              <Card>
                                <Title>{globalActivity.api_base}</Title>
                                <Grid numItems={1}>
                                  <Col>
                                    <Subtitle
                                      style={{
                                        fontSize: "15px",
                                        fontWeight: "normal",
                                        color: "#535452",
                                      }}
                                    >
                                      Num Rate Limit Errors {globalActivity.sum_num_rate_limit_exceptions}
                                    </Subtitle>
                                    <BarChart
                                      className="h-40"
                                      data={globalActivity.daily_data}
                                      index="date"
                                      colors={["rose"]}
                                      categories={["num_rate_limit_exceptions"]}
                                      onValueChange={(v) => console.log(v)}
                                    />
                                  </Col>
                                </Grid>
                              </Card>
                            </Card>
                          ))}
                      </>
                    )}
                  </Grid>
                </TabPanel>
                <TabPanel>
                  <div className="flex items-center gap-4 mb-6">
                    <div className="flex items-center">
                      <Text>Retry Policy Scope:</Text>
                      <Select
                        className="ml-2 w-48"
                        defaultValue="global"
                        value={
                          selectedModelGroup === "global" ? "global" : selectedModelGroup || availableModelGroups[0]
                        }
                        onValueChange={(value) => setSelectedModelGroup(value)}
                      >
                        <SelectItem value="global">Global Default</SelectItem>
                        {availableModelGroups.map((group, idx) => (
                          <SelectItem key={idx} value={group} onClick={() => setSelectedModelGroup(group)}>
                            {group}
                          </SelectItem>
                        ))}
                      </Select>
                    </div>
                  </div>

                  {selectedModelGroup === "global" ? (
                    <>
                      <Title>Global Retry Policy</Title>
                      <Text className="mb-6">Default retry settings applied to all model groups unless overridden</Text>
                    </>
                  ) : (
                    <>
                      <Title>Retry Policy for {selectedModelGroup}</Title>
                      <Text className="mb-6">
                        Model-specific retry settings. Falls back to global defaults if not set.
                      </Text>
                    </>
                  )}
                  {retry_policy_map && (
                    <table>
                      <tbody>
                        {Object.entries(retry_policy_map).map(([exceptionType, retryPolicyKey], idx) => {
                          let retryCount: number;

                          if (selectedModelGroup === "global") {
                            // Show global policy values
                            retryCount = globalRetryPolicy?.[retryPolicyKey] ?? defaultRetry;
                          } else {
                            // Show model-group specific values with fallback to global
                            const modelSpecificCount = modelGroupRetryPolicy?.[selectedModelGroup!]?.[retryPolicyKey];
                            if (modelSpecificCount != null) {
                              retryCount = modelSpecificCount;
                            } else {
                              // Fall back to global policy, then default
                              retryCount = globalRetryPolicy?.[retryPolicyKey] ?? defaultRetry;
                            }
                          }

                          return (
                            <tr key={idx} className="flex justify-between items-center mt-2">
                              <td>
                                <Text>{exceptionType}</Text>
                                {selectedModelGroup !== "global" && (
                                  <Text className="text-xs text-gray-500 ml-2">
                                    (Global: {globalRetryPolicy?.[retryPolicyKey] ?? defaultRetry})
                                  </Text>
                                )}
                              </td>
                              <td>
                                <InputNumber
                                  className="ml-5"
                                  value={retryCount}
                                  min={0}
                                  step={1}
                                  onChange={(value) => {
                                    if (selectedModelGroup === "global") {
                                      // Update global policy
                                      setGlobalRetryPolicy((prevGlobalRetryPolicy) => {
                                        if (value == null) return prevGlobalRetryPolicy;
                                        return {
                                          ...(prevGlobalRetryPolicy ?? {}),
                                          [retryPolicyKey]: value,
                                        };
                                      });
                                    } else {
                                      // Update model-group specific policy
                                      setModelGroupRetryPolicy((prevModelGroupRetryPolicy) => {
                                        const prevRetryPolicy = prevModelGroupRetryPolicy?.[selectedModelGroup!] ?? {};
                                        return {
                                          ...(prevModelGroupRetryPolicy ?? {}),
                                          [selectedModelGroup!]: {
                                            ...prevRetryPolicy,
                                            [retryPolicyKey!]: value,
                                          },
                                        } as RetryPolicyObject;
                                      });
                                    }
                                  }}
                                />
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  )}
                  <Button className="mt-6 mr-8" onClick={handleSaveRetrySettings}>
                    Save
                  </Button>
                </TabPanel>
                <TabPanel>
                  <ModelGroupAliasSettings
                    accessToken={accessToken}
                    initialModelGroupAlias={modelGroupAlias}
                    onAliasUpdate={setModelGroupAlias}
                  />
                </TabPanel>
                <TabPanel>
                  <div className="p-6">
                    <div className="mb-6">
                      <Title>Price Data Management</Title>
                      <Text className="text-tremor-content">
                        Manage model pricing data and configure automatic reload schedules
                      </Text>
                    </div>
                    <PriceDataReload
                      accessToken={accessToken}
                      onReloadSuccess={() => {
                        // Refresh the model map after successful reload
                        const fetchModelMap = async () => {
                          const data = await modelCostMap(accessToken);
                          setModelMap(data);
                        };
                        fetchModelMap();
                      }}
                      buttonText="Reload Price Data"
                      size="middle"
                      type="primary"
                      className="w-full"
                    />
                  </div>
                </TabPanel>
              </TabPanels>
            </TabGroup>
          )}
        </Col>
      </Grid>
    </div>
  );
};

export default OldModelDashboard;
