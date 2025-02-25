import React, { useState, useEffect } from "react";
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
  Metric,
  Text,
  Grid,
  Accordion,
  AccordionHeader,
  AccordionBody,
} from "@tremor/react";


import ConditionalPublicModelName from "./add_model/conditional_public_model_name";
import LiteLLMModelNameField from "./add_model/litellm_model_name";
import AdvancedSettings from "./add_model/advanced_settings";
import ProviderSpecificFields from "./add_model/provider_specific_fields";
import { handleAddModelSubmit } from "./add_model/handle_add_model_submit";
import { getDisplayModelName } from "./view_model/model_name_display";
import EditModelModal, { handleEditModelSubmit } from "./edit_model/edit_model_modal";
import {
  TabPanel,
  TabPanels,
  TabGroup,
  TabList,
  Tab,
  TextInput,
  Icon,
  DateRangePicker,
} from "@tremor/react";
import {
  Select,
  SelectItem,
  MultiSelect,
  MultiSelectItem,
  DateRangePickerValue,
} from "@tremor/react";
import {
  modelInfoCall,
  userGetRequesedtModelsCall,
  modelCreateCall,
  Model,
  modelCostMap,
  modelDeleteCall,
  healthCheckCall,
  modelUpdateCall,
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
} from "./networking";
import { BarChart, AreaChart } from "@tremor/react";
import {
  Button as Button2,
  Modal,
  Popover,
  Form,
  Input,
  Select as AntdSelect,
  InputNumber,
  message,
  Descriptions,
  Tooltip,
  Space,
  Row,
  Col,
} from "antd";
import { Badge, BadgeDelta, Button } from "@tremor/react";
import RequestAccess from "./request_model_access";
import { Typography } from "antd";
import TextArea from "antd/es/input/TextArea";
import {
  InformationCircleIcon,
  PencilAltIcon,
  PencilIcon,
  StatusOnlineIcon,
  TrashIcon,
  RefreshIcon,
  CheckCircleIcon,
  XCircleIcon,
  FilterIcon,
  ChevronUpIcon,
  ChevronDownIcon,
} from "@heroicons/react/outline";
import DeleteModelButton from "./delete_model_button";
const { Title: Title2, Link } = Typography;
import { UploadOutlined } from "@ant-design/icons";
import type { UploadProps } from "antd";
import { Upload } from "antd";
import TimeToFirstToken from "./model_metrics/time_to_first_token";
import DynamicFields from "./model_add/dynamic_form";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { Team } from "./key_team_helpers/key_list";
import TeamInfoView from "./team/team_info";
import { Providers, provider_map, providerLogoMap, getProviderLogoAndName, getPlaceholder, getProviderModels } from "./provider_info_helpers";
import ModelInfoView from "./model_info_view";
import AddModelTab from "./add_model/add_model_tab";
import { ModelDataTable } from "./model_dashboard/table";
import { columns } from "./model_dashboard/columns";

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

interface EditModelModalProps {
  visible: boolean;
  onCancel: () => void;
  model: any; // Assuming TeamType is a type representing your team object
  onSubmit: (data: FormData) => void; // Assuming FormData is the type of data to be submitted
}

interface RetryPolicyObject {
  [key: string]: { [retryPolicyKey: string]: number } | undefined;
}


interface GlobalExceptionActivityData {
  sum_num_rate_limit_exceptions: number;
  daily_data: { date: string; num_rate_limit_exceptions: number; }[];
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

const ModelDashboard: React.FC<ModelDashboardProps> = ({
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
  const [pendingRequests, setPendingRequests] = useState<any[]>([]);
  const [form] = Form.useForm();
  const [modelMap, setModelMap] = useState<any>(null);
  const [lastRefreshed, setLastRefreshed] = useState("");
  
  const [providerModels, setProviderModels] = useState<Array<string>>([]); // Explicitly typing providerModels as a string array

  const providers = Object.values(Providers).filter((key) =>
    isNaN(Number(key))
  );

  const [providerSettings, setProviderSettings] = useState<ProviderSettings[]>(
    []
  );
  const [selectedProvider, setSelectedProvider] = useState<Providers>(Providers.OpenAI);
  const [healthCheckResponse, setHealthCheckResponse] = useState<string>("");
  const [editModalVisible, setEditModalVisible] = useState<boolean>(false);

  const [selectedModel, setSelectedModel] = useState<any>(null);
  const [availableModelGroups, setAvailableModelGroups] = useState<
    Array<string>
  >([]);
  const [availableProviders, setavailableProviders] = useState<
  Array<string>
>([]);
  const [selectedModelGroup, setSelectedModelGroup] = useState<string | null>(
    null
  );
  const [modelLatencyMetrics, setModelLatencyMetrics] = useState<any[]>([]);
  const [modelMetrics, setModelMetrics] = useState<any[]>([]);
  const [modelMetricsCategories, setModelMetricsCategories] = useState<any[]>(
    []
  );
  const [streamingModelMetrics, setStreamingModelMetrics] = useState<any[]>([]);
  const [streamingModelMetricsCategories, setStreamingModelMetricsCategories] =
    useState<any[]>([]);
  const [modelExceptions, setModelExceptions] = useState<any[]>([]);
  const [allExceptions, setAllExceptions] = useState<any[]>([]);
  const [failureTableData, setFailureTableData] = useState<any[]>([]);
  const [slowResponsesData, setSlowResponsesData] = useState<any[]>([]);
  const [dateValue, setDateValue] = useState<DateRangePickerValue>({
    from: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
    to: new Date(),
  });

  const [modelGroupRetryPolicy, setModelGroupRetryPolicy] =
    useState<RetryPolicyObject | null>(null);
  const [defaultRetry, setDefaultRetry] = useState<number>(0);

  const [globalExceptionData, setGlobalExceptionData] =  useState<GlobalExceptionActivityData>({} as GlobalExceptionActivityData);
  const [globalExceptionPerDeployment, setGlobalExceptionPerDeployment] = useState<any[]>([]);

  const [showAdvancedFilters, setShowAdvancedFilters] = useState<boolean>(false);
  const [selectedAPIKey, setSelectedAPIKey] = useState<any | null>(null);
  const [selectedCustomer, setSelectedCustomer] = useState<any | null>(null);

  const [allEndUsers, setAllEndUsers] = useState<any[]>([]);

  // Add state for advanced settings visibility
  const [showAdvancedSettings, setShowAdvancedSettings] = useState<boolean>(false);

  // Add these state variables
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  const [editModel, setEditModel] = useState<boolean>(false);

  const [selectedTeamId, setSelectedTeamId] = useState<string | null>(null);

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
    console.log(
      "inside updateModelMetrics - startTime:",
      startTime,
      "endTime:",
      endTime
    );
    setSelectedModelGroup(modelGroup); // If you want to store the selected model group in state

    let selected_token = selectedAPIKey?.token;
    if (selected_token === undefined) {
      selected_token = null;
    }

    let selected_customer = selectedCustomer;
    if (selected_customer === undefined) {
      selected_customer = null;
    }

    // make startTime and endTime to last hour of the day
    startTime.setHours(0);
    startTime.setMinutes(0);
    startTime.setSeconds(0);

    endTime.setHours(23);
    endTime.setMinutes(59);
    endTime.setSeconds(59);


    try {
      const modelMetricsResponse = await modelMetricsCall(
        accessToken,
        userID,
        userRole,
        modelGroup,
        startTime.toISOString(),
        endTime.toISOString(),
        selected_token,
        selected_customer
      );
      console.log("Model metrics response:", modelMetricsResponse);

      // Assuming modelMetricsResponse now contains the metric data for the specified model group
      setModelMetrics(modelMetricsResponse.data);
      setModelMetricsCategories(modelMetricsResponse.all_api_bases);

      const streamingModelMetricsResponse = await streamingModelMetricsCall(
        accessToken,
        modelGroup,
        startTime.toISOString(),
        endTime.toISOString()
      );

      // Assuming modelMetricsResponse now contains the metric data for the specified model group
      setStreamingModelMetrics(streamingModelMetricsResponse.data);
      setStreamingModelMetricsCategories(
        streamingModelMetricsResponse.all_api_bases
      );

      const modelExceptionsResponse = await modelExceptionsCall(
        accessToken,
        userID,
        userRole,
        modelGroup,
        startTime.toISOString(),
        endTime.toISOString(),
        selected_token,
        selected_customer
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
        selected_customer
      );

      console.log("slowResponses:", slowResponses);

      setSlowResponsesData(slowResponses);


      if (modelGroup) {
        const dailyExceptions = await adminGlobalActivityExceptions(
          accessToken,
          startTime?.toISOString().split('T')[0],
          endTime?.toISOString().split('T')[0],
          modelGroup,
        );

        setGlobalExceptionData(dailyExceptions);

        const dailyExceptionsPerDeplyment = await adminGlobalActivityExceptionsPerDeployment(
          accessToken,
          startTime?.toISOString().split('T')[0],
          endTime?.toISOString().split('T')[0],
          modelGroup,
        )

        setGlobalExceptionPerDeployment(dailyExceptionsPerDeplyment);

      }

      
    } catch (error) {
      console.error("Failed to fetch model metrics", error);
    }
  };


  useEffect(() => {
    updateModelMetrics(
      selectedModelGroup,
      dateValue.from,
      dateValue.to
    );
  }, [selectedAPIKey, selectedCustomer]);

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
            form.setFieldsValue({ vertex_credentials: jsonStr });
          }
        };
        reader.readAsText(file);
      }
      // Prevent upload
      return false;
    },
    onChange(info) {
      if (info.file.status !== "uploading") {
        console.log(info.file, info.fileList);
      }
      if (info.file.status === "done") {
        message.success(`${info.file.name} file uploaded successfully`);
      } else if (info.file.status === "error") {
        message.error(`${info.file.name} file upload failed.`);
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

    console.log("new modelGroupRetryPolicy:", modelGroupRetryPolicy);

    try {
      const payload = {
        router_settings: {
          model_group_retry_policy: modelGroupRetryPolicy,
        },
      };

      await setCallbacksCall(accessToken, payload);
      message.success("Retry settings saved successfully");
    } catch (error) {
      console.error("Failed to save retry settings:", error);
      message.error("Failed to save retry settings");
    }
  };

  useEffect(() => {
    if (!accessToken || !token || !userRole || !userID) {
      return;
    }
    const fetchData = async () => {
      try {
        const _providerSettings = await modelSettingsCall(accessToken);
        setProviderSettings(_providerSettings);

        // Replace with your actual API call for model data
        const modelDataResponse = await modelInfoCall(
          accessToken,
          userID,
          userRole
        );
        console.log("Model data response:", modelDataResponse.data);
        setModelData(modelDataResponse);

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

        console.log("array_model_groups:", _array_model_groups);
        let _initial_model_group = "all";
        if (_array_model_groups.length > 0) {
          // set selectedModelGroup to the last model group
          _initial_model_group =
            _array_model_groups[_array_model_groups.length - 1];
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
          selectedCustomer
        );

        console.log("Model metrics response:", modelMetricsResponse);
        // Sort by latency (avg_latency_per_token)

        setModelMetrics(modelMetricsResponse.data);
        setModelMetricsCategories(modelMetricsResponse.all_api_bases);

        const streamingModelMetricsResponse = await streamingModelMetricsCall(
          accessToken,
          _initial_model_group,
          dateValue.from?.toISOString(),
          dateValue.to?.toISOString()
        );

        // Assuming modelMetricsResponse now contains the metric data for the specified model group
        setStreamingModelMetrics(streamingModelMetricsResponse.data);
        setStreamingModelMetricsCategories(
          streamingModelMetricsResponse.all_api_bases
        );

        const modelExceptionsResponse = await modelExceptionsCall(
          accessToken,
          userID,
          userRole,
          _initial_model_group,
          dateValue.from?.toISOString(),
          dateValue.to?.toISOString(),
          selectedAPIKey?.token,
          selectedCustomer
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
          selectedCustomer
        );

        const dailyExceptions = await adminGlobalActivityExceptions(
          accessToken,
          dateValue.from?.toISOString().split('T')[0],
          dateValue.to?.toISOString().split('T')[0],
          _initial_model_group,
        );

        setGlobalExceptionData(dailyExceptions);

        const dailyExceptionsPerDeplyment = await adminGlobalActivityExceptionsPerDeployment(
          accessToken,
          dateValue.from?.toISOString().split('T')[0],
          dateValue.to?.toISOString().split('T')[0],
          _initial_model_group,
        )

        setGlobalExceptionPerDeployment(dailyExceptionsPerDeplyment);

        console.log("dailyExceptions:", dailyExceptions);

        console.log("dailyExceptionsPerDeplyment:", dailyExceptionsPerDeplyment);

      
        console.log("slowResponses:", slowResponses);

        setSlowResponsesData(slowResponses);

        let all_end_users_data = await allEndUsersCall(accessToken);

        setAllEndUsers(all_end_users_data?.end_users);

        const routerSettingsInfo = await getCallbacksCall(
          accessToken,
          userID,
          userRole
        );

        let router_settings = routerSettingsInfo.router_settings;

        console.log("routerSettingsInfo:", router_settings);

        let model_group_retry_policy = router_settings.model_group_retry_policy;
        let default_retries = router_settings.num_retries;

        console.log("model_group_retry_policy:", model_group_retry_policy);
        console.log("default_retries:", default_retries);
        setModelGroupRetryPolicy(model_group_retry_policy);
        setDefaultRetry(default_retries);
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
  }, [accessToken, token, userRole, userID, modelMap, lastRefreshed]);

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
        provider =
        splitModel.length === 1
          ? getProviderFromModel(litellm_model_name)
          : firstElement;
        
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
        Object.entries(curr_model?.litellm_params).filter(
          ([key]) => key !== "model" && key !== "api_base"
        )
      );
    }

    modelData.data[i].provider = provider;
    modelData.data[i].input_cost = input_cost;
    modelData.data[i].output_cost = output_cost;
    modelData.data[i].litellm_model_name = litellm_model_name;
    all_providers.push(provider);

    // Convert Cost in terms of Cost per 1M tokens
    if (modelData.data[i].input_cost) {
      modelData.data[i].input_cost = (
        Number(modelData.data[i].input_cost) * 1000000
      ).toFixed(2);
    }

    if (modelData.data[i].output_cost) {
      modelData.data[i].output_cost = (
        Number(modelData.data[i].output_cost) * 1000000
      ).toFixed(2);
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
        <Paragraph>
          Ask your proxy admin for access to view all models
        </Paragraph>
      </div>
    );
  }

  const runHealthCheck = async () => {
    try {
      message.info("Running health check...");
      setHealthCheckResponse("");
      const response = await healthCheckCall(accessToken);
      setHealthCheckResponse(response);
    } catch (error) {
      console.error("Error running health check:", error);
      setHealthCheckResponse("Error running health check");
    }
  };

  const FilterByContent = (
      <div >
        <Text className="mb-1">Select API Key Name</Text>

        {
          premiumUser ? (
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
                      if (
                        key &&
                        key["key_alias"] !== null &&
                        key["key_alias"].length > 0
                      ) {
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
                      return null; // Add this line to handle the case when the condition is not met
                    })}
                  </Select>
          

          <Text className="mt-1">
            Select Customer Name
          </Text>
          
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
            {
              allEndUsers?.map((user: any, index: number) => {
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
              })
            }
          </Select>
            
            </div>
          ): (
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
                      if (
                        key &&
                        key["key_alias"] !== null &&
                        key["key_alias"].length > 0
                      ) {
                        return (
                          
                          <SelectItem
                            key={index}
                            value={String(index)}
                            // @ts-ignore
                            disabled={true}
                            onClick={() => {
                              setSelectedAPIKey(key);
                            }}
                          >
                            ✨ {key["key_alias"]} (Enterprise only Feature)
                          </SelectItem>
                        );
                      }
                      return null; // Add this line to handle the case when the condition is not met
                    })}
                  </Select>
          

          <Text className="mt-1">
            Select Customer Name
          </Text>
          
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
            {
              allEndUsers?.map((user: any, index: number) => {
                return (
                  <SelectItem
                    key={index}
                    value={user}
                    // @ts-ignore
                    disabled={true}
                    onClick={() => {
                      setSelectedCustomer(user);
                    }}
                  >
                    ✨ {user} (Enterprise only Feature)
                  </SelectItem>
                );
              })
            }
          </Select>
            
            </div>
          )
        }
        

        
            

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
        value: payload
          .slice(5)
          .reduce((acc: number, curr: any) => acc + curr.value, 0),
        color: "gray",
      });
    }

    return (
      <div className="w-150 rounded-tremor-default border border-tremor-border bg-tremor-background p-2 text-tremor-default shadow-tremor-dropdown">
        {date && (
          <p className="text-tremor-content-emphasis mb-2">Date: {date}</p>
        )}
        {sortedPayload.map((category: any, idx: number) => {
          const roundedValue = parseFloat(category.value.toFixed(5));
          const displayValue =
            roundedValue === 0 && category.value > 0
              ? "<0.00001"
              : roundedValue.toFixed(5);
          return (
            <div key={idx} className="flex justify-between">
              <div className="flex items-center space-x-2">
                <div
                  className={`w-2 h-2 mt-1 rounded-full bg-${category.color}-500`}
                />
                <p className="text-tremor-content">{category.dataKey}</p>
              </div>
              <p className="font-medium text-tremor-content-emphasis text-righ ml-2">
                {displayValue}
              </p>
            </div>
          );
        })}
      </div>
    );
  };


  const handleOk = () => {
    form
      .validateFields()
      .then((values) => {
        handleAddModelSubmit(values, accessToken, form, handleRefreshClick);
      })
      .catch((error) => {
        console.error("Validation failed:", error);
      });
  };

  console.log(`selectedProvider: ${selectedProvider}`);
  console.log(`providerModels.length: ${providerModels.length}`);

  const providerKey = Object.keys(Providers).find(
    (key) => (Providers as { [index: string]: any })[key] === selectedProvider
  );

  let dynamicProviderForm: ProviderSettings | undefined = undefined;
  if (providerKey) {
    dynamicProviderForm = providerSettings.find(
      (provider) => provider.name === provider_map[providerKey]
    );
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
        />
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height: "100%" }}>
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
        />
      ) : (
        <TabGroup className="gap-2 p-8 h-[75vh] w-full mt-2">
          <TabList className="flex justify-between mt-2 w-full items-center">
            <div className="flex">
              <Tab>All Models</Tab>
              <Tab>Add Model</Tab>
              <Tab>
                <pre>/health Models</pre>
              </Tab>
              <Tab>Model Analytics</Tab>
              <Tab>Model Retry Settings</Tab>
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
                <div className="flex items-center">
                  <Text>Filter by Public Model Name</Text>
                  <Select
                    className="mb-4 mt-2 ml-2 w-50"
                    defaultValue={
                      selectedModelGroup
                        ? selectedModelGroup
                        : undefined
                    }
                    onValueChange={(value) =>
                      setSelectedModelGroup(value === "all" ? "all" : value)
                    }
                    value={
                      selectedModelGroup
                        ? selectedModelGroup
                        : undefined
                    }
                  >
                    <SelectItem value={"all"}>All Models</SelectItem>
                    {availableModelGroups.map((group, idx) => (
                      <SelectItem
                        key={idx}
                        value={group}
                        onClick={() => setSelectedModelGroup(group)}
                      >
                        {group}
                      </SelectItem>
                    ))}
                  </Select>
                </div>
                  <ModelDataTable
                    columns={columns(
                      premiumUser,
                      setSelectedModelId,
                      setSelectedTeamId,
                      getDisplayModelName,
                      handleEditClick,
                      handleRefreshClick,
                      setEditModel
                    )}
                    data={modelData.data.filter(
                      (model: any) =>
                        selectedModelGroup === "all" ||
                        model.model_name === selectedModelGroup ||
                        !selectedModelGroup
                    )}
                    isLoading={false} // Add loading state if needed
                  />
              </Grid>
              <EditModelModal
                visible={editModalVisible}
                onCancel={handleEditCancel}
                model={selectedModel}
                onSubmit={(data: FormData) => handleEditModelSubmit(data, accessToken, setEditModalVisible, setSelectedModel)}
              />
            </TabPanel>
            <TabPanel className="h-full">
              <AddModelTab
                form={form}
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
              />
            </TabPanel>
            <TabPanel>
              <Card>
                <Text>
                  `/health` will run a very small request through your models
                  configured on litellm
                </Text>

                <Button onClick={runHealthCheck}>Run `/health`</Button>
                {healthCheckResponse && (
                  <pre>{JSON.stringify(healthCheckResponse, null, 2)}</pre>
                )}
              </Card>
            </TabPanel>
            <TabPanel>
              <div className="flex items-center">
                <Text>Filter by Public Model Name</Text>

                <Select
                  className="mb-4 mt-2 ml-2 w-50"
                  defaultValue={
                    selectedModelGroup
                      ? selectedModelGroup
                      : availableModelGroups[0]
                  }
                  value={
                    selectedModelGroup
                      ? selectedModelGroup
                      : availableModelGroups[0]
                  }
                  onValueChange={(value) => setSelectedModelGroup(value)}
                >
                  {availableModelGroups.map((group, idx) => (
                    <SelectItem
                      key={idx}
                      value={group}
                      onClick={() => setSelectedModelGroup(group)}
                    >
                      {group}
                    </SelectItem>
                  ))}
                </Select>
              </div>

              <Title>Retry Policy for {selectedModelGroup}</Title>
              <Text className="mb-6">
                How many retries should be attempted based on the Exception
              </Text>
              {retry_policy_map && (
                <table>
                  <tbody>
                    {Object.entries(retry_policy_map).map(
                      ([exceptionType, retryPolicyKey], idx) => {
                        let retryCount =
                          modelGroupRetryPolicy?.[selectedModelGroup!]?.[
                            retryPolicyKey
                          ];
                        if (retryCount == null) {
                          retryCount = defaultRetry;
                        }

                        return (
                          <tr
                            key={idx}
                            className="flex justify-between items-center mt-2"
                          >
                            <td>
                              <Text>{exceptionType}</Text>
                            </td>
                            <td>
                              <InputNumber
                                className="ml-5"
                                value={retryCount}
                                min={0}
                                step={1}
                                onChange={(value) => {
                                  setModelGroupRetryPolicy(
                                    (prevModelGroupRetryPolicy) => {
                                      const prevRetryPolicy =
                                        prevModelGroupRetryPolicy?.[
                                          selectedModelGroup!
                                        ] ?? {};
                                      return {
                                        ...(prevModelGroupRetryPolicy ?? {}),
                                        [selectedModelGroup!]: {
                                          ...prevRetryPolicy,
                                          [retryPolicyKey!]: value,
                                        },
                                      } as RetryPolicyObject;
                                    }
                                  );
                                }}
                              />
                            </td>
                          </tr>
                        );
                      }
                    )}
                  </tbody>
                </table>
              )}
              <Button className="mt-6 mr-8" onClick={handleSaveRetrySettings}>
                Save
              </Button>
            </TabPanel>
          </TabPanels>
        </TabGroup>
      )}
    </div>  
  );
};

export default ModelDashboard;
