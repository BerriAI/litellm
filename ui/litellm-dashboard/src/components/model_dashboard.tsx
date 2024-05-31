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
} from "./networking";
import { BarChart, AreaChart } from "@tremor/react";
import {
  Button as Button2,
  Modal,
  Form,
  Input,
  Select as Select2,
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
} from "@heroicons/react/outline";
import DeleteModelButton from "./delete_model_button";
const { Title: Title2, Link } = Typography;
import { UploadOutlined } from "@ant-design/icons";
import type { UploadProps } from "antd";
import { Upload } from "antd";
import TimeToFirstToken from "./model_metrics/time_to_first_token";
import DynamicFields from "./model_add/dynamic_form";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";

interface ModelDashboardProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  modelData: any;
  setModelData: any;
  premiumUser: boolean;
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

enum Providers {
  OpenAI = "OpenAI",
  Azure = "Azure",
  Anthropic = "Anthropic",
  Google_AI_Studio = "Google AI Studio",
  Bedrock = "Amazon Bedrock",
  OpenAI_Compatible = "OpenAI-Compatible Endpoints (Groq, Together AI, Mistral AI, etc.)",
  Vertex_AI = "Vertex AI (Anthropic, Gemini, etc.)",
  Databricks = "Databricks",
}

const provider_map: Record<string, string> = {
  OpenAI: "openai",
  Azure: "azure",
  Anthropic: "anthropic",
  Google_AI_Studio: "gemini",
  Bedrock: "bedrock",
  OpenAI_Compatible: "openai",
  Vertex_AI: "vertex_ai",
  Databricks: "databricks",
};

const retry_policy_map: Record<string, string> = {
  "BadRequestError (400)": "BadRequestErrorRetries",
  "AuthenticationError  (401)": "AuthenticationErrorRetries",
  "TimeoutError (408)": "TimeoutErrorRetries",
  "RateLimitError (429)": "RateLimitErrorRetries",
  "ContentPolicyViolationError (400)": "ContentPolicyViolationErrorRetries",
  "InternalServerError (500)": "InternalServerErrorRetries",
};

const handleSubmit = async (
  formValues: Record<string, any>,
  accessToken: string,
  form: any
) => {
  try {
    /**
     * For multiple litellm model names - create a separate deployment for each
     * - get the list
     * - iterate through it
     * - create a new deployment for each
     *
     * For single model name -> make it a 1 item list
     */

    // get the list of deployments
    let deployments: Array<string> = Array.isArray(formValues["model"])
      ? formValues["model"]
      : [formValues["model"]];
    console.log(`received deployments: ${deployments}`);
    console.log(`received type of deployments: ${typeof deployments}`);
    deployments.forEach(async (litellm_model) => {
      console.log(`litellm_model: ${litellm_model}`);
      const litellmParamsObj: Record<string, any> = {};
      const modelInfoObj: Record<string, any> = {};
      // Iterate through the key-value pairs in formValues
      litellmParamsObj["model"] = litellm_model;
      let modelName: string = "";
      console.log("formValues add deployment:", formValues);
      for (const [key, value] of Object.entries(formValues)) {
        if (value === "") {
          continue;
        }
        if (key == "model_name") {
          modelName = modelName + value;
        } else if (key == "custom_llm_provider") {
          // const providerEnumValue = Providers[value as keyof typeof Providers];
          // const mappingResult = provider_map[providerEnumValue]; // Get the corresponding value from the mapping
          // modelName = mappingResult + "/" + modelName
          continue;
        } else if (key == "model") {
          continue;
        }

        // Check if key is "base_model"
        else if (key === "base_model") {
          // Add key-value pair to model_info dictionary
          modelInfoObj[key] = value;
        } else if (key == "litellm_extra_params") {
          console.log("litellm_extra_params:", value);
          let litellmExtraParams = {};
          if (value && value != undefined) {
            try {
              litellmExtraParams = JSON.parse(value);
            } catch (error) {
              message.error(
                "Failed to parse LiteLLM Extra Params: " + error,
                10
              );
              throw new Error("Failed to parse litellm_extra_params: " + error);
            }
            for (const [key, value] of Object.entries(litellmExtraParams)) {
              litellmParamsObj[key] = value;
            }
          }
        }

        // Check if key is any of the specified API related keys
        else {
          // Add key-value pair to litellm_params dictionary
          litellmParamsObj[key] = value;
        }
      }

      const new_model: Model = {
        model_name: modelName,
        litellm_params: litellmParamsObj,
        model_info: modelInfoObj,
      };

      const response: any = await modelCreateCall(accessToken, new_model);

      console.log(`response for model create call: ${response["data"]}`);
    });

    form.resetFields();
  } catch (error) {
    message.error("Failed to create model: " + error, 10);
  }
};

const ModelDashboard: React.FC<ModelDashboardProps> = ({
  accessToken,
  token,
  userRole,
  userID,
  modelData = { data: [] },
  setModelData,
  premiumUser,
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
  const [selectedProvider, setSelectedProvider] = useState<String>("OpenAI");
  const [healthCheckResponse, setHealthCheckResponse] = useState<string>("");
  const [editModalVisible, setEditModalVisible] = useState<boolean>(false);
  const [infoModalVisible, setInfoModalVisible] = useState<boolean>(false);

  const [selectedModel, setSelectedModel] = useState<any>(null);
  const [availableModelGroups, setAvailableModelGroups] = useState<
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

  function formatCreatedAt(createdAt: string | null) {
    if (createdAt) {
      const date = new Date(createdAt);
      const options = { month: "long", day: "numeric", year: "numeric" };
      return date.toLocaleDateString("en-US");
    }
    return null;
  }

  const EditModelModal: React.FC<EditModelModalProps> = ({
    visible,
    onCancel,
    model,
    onSubmit,
  }) => {
    const [form] = Form.useForm();
    let litellm_params_to_edit: Record<string, any> = {};
    let model_name = "";
    let model_id = "";
    if (model) {
      litellm_params_to_edit = model.litellm_params;
      model_name = model.model_name;
      let model_info = model.model_info;
      if (model_info) {
        model_id = model_info.id;
        console.log(`model_id: ${model_id}`);
        litellm_params_to_edit.model_id = model_id;
      }
    }

    const handleOk = () => {
      form
        .validateFields()
        .then((values) => {
          onSubmit(values);
          form.resetFields();
        })
        .catch((error) => {
          console.error("Validation failed:", error);
        });
    };

    return (
      <Modal
        title={"Edit Model " + model_name}
        visible={visible}
        width={800}
        footer={null}
        onOk={handleOk}
        onCancel={onCancel}
      >
        <Form
          form={form}
          onFinish={handleEditSubmit}
          initialValues={litellm_params_to_edit} // Pass initial values here
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          labelAlign="left"
        >
          <>
            <Form.Item className="mt-8" label="api_base" name="api_base">
              <TextInput />
            </Form.Item>

            <Form.Item
              label="tpm"
              name="tpm"
              tooltip="int (optional) - Tokens limit for this deployment: in tokens per minute (tpm). Find this information on your model/providers website"
            >
              <InputNumber min={0} step={1} />
            </Form.Item>

            <Form.Item
              label="rpm"
              name="rpm"
              tooltip="int (optional) - Rate limit for this deployment: in requests per minute (rpm). Find this information on your model/providers website"
            >
              <InputNumber min={0} step={1} />
            </Form.Item>

            <Form.Item label="max_retries" name="max_retries">
              <InputNumber min={0} step={1} />
            </Form.Item>

            <Form.Item
              label="timeout"
              name="timeout"
              tooltip="int (optional) - Timeout in seconds for LLM requests (Defaults to 600 seconds)"
            >
              <InputNumber min={0} step={1} />
            </Form.Item>

            <Form.Item
              label="stream_timeout"
              name="stream_timeout"
              tooltip="int (optional) - Timeout for stream requests (seconds)"
            >
              <InputNumber min={0} step={1} />
            </Form.Item>

            <Form.Item
              label="input_cost_per_token"
              name="input_cost_per_token"
              tooltip="float (optional) - Input cost per token"
            >
              <InputNumber min={0} step={0.0001} />
            </Form.Item>

            <Form.Item
              label="output_cost_per_token"
              name="output_cost_per_token"
              tooltip="float (optional) - Output cost per token"
            >
              <InputNumber min={0} step={0.0001} />
            </Form.Item>

            <Form.Item
              label="model_id"
              name="model_id"
              hidden={true}
            ></Form.Item>
          </>
          <div style={{ textAlign: "right", marginTop: "10px" }}>
            <Button2 htmlType="submit">Save</Button2>
          </div>
        </Form>
      </Modal>
    );
  };

  const handleEditClick = (model: any) => {
    setSelectedModel(model);
    setEditModalVisible(true);
  };

  const handleInfoClick = (model: any) => {
    setSelectedModel(model);
    setInfoModalVisible(true);
  };

  const handleEditCancel = () => {
    setEditModalVisible(false);
    setSelectedModel(null);
  };

  const handleInfoCancel = () => {
    setInfoModalVisible(false);
    setSelectedModel(null);
  };

  const handleEditSubmit = async (formValues: Record<string, any>) => {
    // Call API to update team with teamId and values

    console.log("handleEditSubmit:", formValues);
    if (accessToken == null) {
      return;
    }

    let newLiteLLMParams: Record<string, any> = {};
    let model_info_model_id = null;

    for (const [key, value] of Object.entries(formValues)) {
      if (key !== "model_id") {
        newLiteLLMParams[key] = value;
      } else {
        model_info_model_id = value;
      }
    }

    let payload = {
      litellm_params: newLiteLLMParams,
      model_info: {
        id: model_info_model_id,
      },
    };

    console.log("handleEditSubmit payload:", payload);

    try {
      let newModelValue = await modelUpdateCall(accessToken, payload);
      message.success(
        "Model updated successfully, restart server to see updates"
      );

      setEditModalVisible(false);
      setSelectedModel(null);
    } catch (error) {
      console.log(`Error occurred`);
    }
  };

  const props: UploadProps = {
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
          setSelectedModelGroup(_initial_model_group);
        }

        console.log("selectedModelGroup:", selectedModelGroup);

        const modelMetricsResponse = await modelMetricsCall(
          accessToken,
          userID,
          userRole,
          _initial_model_group,
          dateValue.from?.toISOString(),
          dateValue.to?.toISOString()
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
          dateValue.to?.toISOString()
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
          dateValue.to?.toISOString()
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
      const data = await modelCostMap();
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

  // loop through model data and edit each row
  for (let i = 0; i < modelData.data.length; i++) {
    let curr_model = modelData.data[i];
    let litellm_model_name = curr_model?.litellm_params?.model;
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
      provider =
        splitModel.length === 1
          ? getProviderFromModel(litellm_model_name)
          : firstElement;
    } else {
      // litellm_model_name is null or undefined, default provider to openai
      provider = "openai";
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

  const setProviderModelsFn = (provider: string) => {
    console.log(`received provider string: ${provider}`);
    const providerKey = Object.keys(Providers).find(
      (key) => (Providers as { [index: string]: any })[key] === provider
    );

    if (providerKey) {
      const mappingResult = provider_map[providerKey]; // Get the corresponding value from the mapping
      console.log(`mappingResult: ${mappingResult}`);
      let _providerModels: Array<string> = [];
      if (typeof modelMap === "object") {
        Object.entries(modelMap).forEach(([key, value]) => {
          if (
            value !== null &&
            typeof value === "object" &&
            "litellm_provider" in (value as object) &&
            ((value as any)["litellm_provider"] === mappingResult ||
              (value as any)["litellm_provider"].includes(mappingResult))
          ) {
            _providerModels.push(key);
          }
        });
      }
      setProviderModels(_providerModels);
      console.log(`providerModels: ${providerModels}`);
    }
  };

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

  const updateModelMetrics = async (
    modelGroup: string | null,
    startTime: Date | undefined,
    endTime: Date | undefined
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

    try {
      const modelMetricsResponse = await modelMetricsCall(
        accessToken,
        userID,
        userRole,
        modelGroup,
        startTime.toISOString(),
        endTime.toISOString()
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
        endTime.toISOString()
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
        endTime.toISOString()
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

  const getPlaceholder = (selectedProvider: string): string => {
    if (selectedProvider === Providers.Vertex_AI) {
      return "gemini-pro";
    } else if (selectedProvider == Providers.Anthropic) {
      return "claude-3-opus";
    } else if (selectedProvider == Providers.Bedrock) {
      return "claude-3-opus";
    } else if (selectedProvider == Providers.Google_AI_Studio) {
      return "gemini-pro";
    } else {
      return "gpt-3.5-turbo";
    }
  };

  const handleOk = () => {
    form
      .validateFields()
      .then((values) => {
        handleSubmit(values, accessToken, form);
        // form.resetFields();
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

  return (
    <div style={{ width: "100%", height: "100%" }}>
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
                      : availableModelGroups[0]
                  }
                  onValueChange={(value) =>
                    setSelectedModelGroup(value === "all" ? "all" : value)
                  }
                  value={
                    selectedModelGroup
                      ? selectedModelGroup
                      : availableModelGroups[0]
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
              <Card>
                <Table style={{ maxWidth: "1500px", width: "100%" }}>
                  <TableHead>
                    <TableRow>
                      <TableHeaderCell
                        style={{
                          maxWidth: "150px",
                          whiteSpace: "normal",
                          wordBreak: "break-word",
                          fontSize: "11px",
                        }}
                      >
                        Public Model Name
                      </TableHeaderCell>
                      <TableHeaderCell
                        style={{
                          maxWidth: "100px",
                          whiteSpace: "normal",
                          wordBreak: "break-word",
                          fontSize: "11px",
                        }}
                      >
                        Provider
                      </TableHeaderCell>
                      {userRole === "Admin" && (
                        <TableHeaderCell
                          style={{
                            maxWidth: "150px",
                            whiteSpace: "normal",
                            wordBreak: "break-word",
                            fontSize: "11px",
                          }}
                        >
                          API Base
                        </TableHeaderCell>
                      )}
                      <TableHeaderCell
                        style={{
                          maxWidth: "85px",
                          whiteSpace: "normal",
                          wordBreak: "break-word",
                          fontSize: "11px",
                        }}
                      >
                        Input Price{" "}
                        <p style={{ fontSize: "10px", color: "gray" }}>
                          /1M Tokens ($)
                        </p>
                      </TableHeaderCell>
                      <TableHeaderCell
                        style={{
                          maxWidth: "85px",
                          whiteSpace: "normal",
                          wordBreak: "break-word",
                          fontSize: "11px",
                        }}
                      >
                        Output Price{" "}
                        <p style={{ fontSize: "10px", color: "gray" }}>
                          /1M Tokens ($)
                        </p>
                      </TableHeaderCell>

                      <TableHeaderCell
                        style={{
                          maxWidth: "100px",
                          whiteSpace: "normal",
                          wordBreak: "break-word",
                          fontSize: "11px",
                        }}
                      >
                        {premiumUser ? (
                          "Created At"
                        ) : (
                          <a
                            href="https://forms.gle/W3U4PZpJGFHWtHyA9"
                            target="_blank"
                            style={{ color: "#72bcd4" }}
                          >
                            {" "}
                            ✨ Created At
                          </a>
                        )}
                      </TableHeaderCell>
                      <TableHeaderCell
                        style={{
                          maxWidth: "100px",
                          whiteSpace: "normal",
                          wordBreak: "break-word",
                          fontSize: "11px",
                        }}
                      >
                        {premiumUser ? (
                          "Created By"
                        ) : (
                          <a
                            href="https://forms.gle/W3U4PZpJGFHWtHyA9"
                            target="_blank"
                            style={{ color: "#72bcd4" }}
                          >
                            {" "}
                            ✨ Created By
                          </a>
                        )}
                      </TableHeaderCell>
                      <TableHeaderCell
                        style={{
                          maxWidth: "50px",
                          whiteSpace: "normal",
                          wordBreak: "break-word",
                          fontSize: "11px",
                        }}
                      >
                        Status
                      </TableHeaderCell>
                      <TableHeaderCell></TableHeaderCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {modelData.data
                      .filter(
                        (model: any) =>
                          selectedModelGroup === "all" ||
                          model.model_name === selectedModelGroup ||
                          selectedModelGroup === null ||
                          selectedModelGroup === undefined ||
                          selectedModelGroup === ""
                      )
                      .map((model: any, index: number) => (
                        <TableRow
                          key={index}
                          style={{ maxHeight: "1px", minHeight: "1px" }}
                        >
                          <TableCell
                            style={{
                              maxWidth: "100px",
                              whiteSpace: "normal",
                              wordBreak: "break-word",
                            }}
                          >
                            <p style={{ fontSize: "10px" }}>
                              {model.model_name || "-"}
                            </p>
                          </TableCell>
                          <TableCell
                            style={{
                              maxWidth: "100px",
                              whiteSpace: "normal",
                              wordBreak: "break-word",
                            }}
                          >
                            <p style={{ fontSize: "10px" }}>
                              {model.provider || "-"}
                            </p>
                          </TableCell>
                          {userRole === "Admin" && (
                            <TableCell
                              style={{
                                maxWidth: "150px",
                                whiteSpace: "normal",
                                wordBreak: "break-word",
                              }}
                            >
                              <Tooltip title={model && model.api_base}>
                                <pre
                                  style={{
                                    maxWidth: "150px",
                                    whiteSpace: "normal",
                                    wordBreak: "break-word",
                                    fontSize: "10px",
                                  }}
                                  title={
                                    model && model.api_base
                                      ? model.api_base
                                      : ""
                                  }
                                >
                                  {model && model.api_base
                                    ? model.api_base.slice(0, 20)
                                    : "-"}
                                </pre>
                              </Tooltip>
                            </TableCell>
                          )}
                          <TableCell
                            style={{
                              maxWidth: "80px",
                              whiteSpace: "normal",
                              wordBreak: "break-word",
                            }}
                          >
                            <pre style={{ fontSize: "10px" }}>
                              {model.input_cost
                                ? model.input_cost
                                : model.litellm_params.input_cost_per_token
                                  ? (
                                      Number(
                                        model.litellm_params
                                          .input_cost_per_token
                                      ) * 1000000
                                    ).toFixed(2)
                                  : null}
                            </pre>
                          </TableCell>
                          <TableCell
                            style={{
                              maxWidth: "80px",
                              whiteSpace: "normal",
                              wordBreak: "break-word",
                            }}
                          >
                            <pre style={{ fontSize: "10px" }}>
                              {model.output_cost
                                ? model.output_cost
                                : model.litellm_params.output_cost_per_token
                                  ? (
                                      Number(
                                        model.litellm_params
                                          .output_cost_per_token
                                      ) * 1000000
                                    ).toFixed(2)
                                  : null}
                            </pre>
                          </TableCell>
                          <TableCell>
                            <p style={{ fontSize: "10px" }}>
                              {premiumUser
                                ? formatCreatedAt(
                                    model.model_info.created_at
                                  ) || "-"
                                : "-"}
                            </p>
                          </TableCell>
                          <TableCell>
                            <p style={{ fontSize: "10px" }}>
                              {premiumUser
                                ? model.model_info.created_by || "-"
                                : "-"}
                            </p>
                          </TableCell>
                          <TableCell
                            style={{
                              maxWidth: "100px",
                              whiteSpace: "normal",
                              wordBreak: "break-word",
                            }}
                          >
                            {model.model_info.db_model ? (
                              <Badge size="xs" className="text-white">
                                <p style={{ fontSize: "10px" }}>DB Model</p>
                              </Badge>
                            ) : (
                              <Badge size="xs" className="text-black">
                                <p style={{ fontSize: "10px" }}>Config Model</p>
                              </Badge>
                            )}
                          </TableCell>
                          <TableCell
                            style={{
                              maxWidth: "150px",
                              whiteSpace: "normal",
                              wordBreak: "break-word",
                            }}
                          >
                            <Grid numItems={3}>
                              <Col>
                                <Icon
                                  icon={InformationCircleIcon}
                                  size="sm"
                                  onClick={() => handleInfoClick(model)}
                                />
                              </Col>
                              <Col>
                                <Icon
                                  icon={PencilAltIcon}
                                  size="sm"
                                  onClick={() => handleEditClick(model)}
                                />
                              </Col>

                              <Col>
                                <DeleteModelButton
                                  modelID={model.model_info.id}
                                  accessToken={accessToken}
                                />
                              </Col>
                            </Grid>
                          </TableCell>
                        </TableRow>
                      ))}
                  </TableBody>
                </Table>
              </Card>
            </Grid>
            <EditModelModal
              visible={editModalVisible}
              onCancel={handleEditCancel}
              model={selectedModel}
              onSubmit={handleEditSubmit}
            />
            <Modal
              title={selectedModel && selectedModel.model_name}
              visible={infoModalVisible}
              width={800}
              footer={null}
              onCancel={handleInfoCancel}
            >
              <Title>Model Info</Title>
              <SyntaxHighlighter language="json">
                {selectedModel && JSON.stringify(selectedModel, null, 2)}
              </SyntaxHighlighter>
            </Modal>
          </TabPanel>
          <TabPanel className="h-full">
            <Title2 level={2}>Add new model</Title2>
            <Card>
              <Form
                form={form}
                onFinish={handleOk}
                labelCol={{ span: 10 }}
                wrapperCol={{ span: 16 }}
                labelAlign="left"
              >
                <>
                  <Form.Item
                    rules={[{ required: true, message: "Required" }]}
                    label="Provider:"
                    name="custom_llm_provider"
                    tooltip="E.g. OpenAI, Azure OpenAI, Anthropic, Bedrock, etc."
                    labelCol={{ span: 10 }}
                    labelAlign="left"
                  >
                    <Select value={selectedProvider.toString()}>
                      {providers.map((provider, index) => (
                        <SelectItem
                          key={index}
                          value={provider}
                          onClick={() => {
                            setProviderModelsFn(provider);
                            setSelectedProvider(provider);
                          }}
                        >
                          {provider}
                        </SelectItem>
                      ))}
                    </Select>
                  </Form.Item>

                  <Form.Item
                    rules={[{ required: true, message: "Required" }]}
                    label="Public Model Name"
                    name="model_name"
                    tooltip="Model name your users will pass in. Also used for load-balancing, LiteLLM will load balance between all models with this public name."
                    className="mb-0"
                  >
                    <TextInput
                      placeholder={getPlaceholder(selectedProvider.toString())}
                    />
                  </Form.Item>
                  <Row>
                    <Col span={10}></Col>
                    <Col span={10}>
                      <Text className="mb-3 mt-1">
                        Model name your users will pass in.
                      </Text>
                    </Col>
                  </Row>
                  <Form.Item
                    rules={[{ required: true, message: "Required" }]}
                    label="LiteLLM Model Name(s)"
                    name="model"
                    tooltip="Actual model name used for making litellm.completion() call."
                    className="mb-0"
                  >
                    {selectedProvider === Providers.Azure ? (
                      <TextInput placeholder="Enter model name" />
                    ) : providerModels.length > 0 ? (
                      <MultiSelect value={providerModels}>
                        {providerModels.map((model, index) => (
                          <MultiSelectItem key={index} value={model}>
                            {model}
                          </MultiSelectItem>
                        ))}
                      </MultiSelect>
                    ) : (
                      <TextInput placeholder="gpt-3.5-turbo-0125" />
                    )}
                  </Form.Item>
                  <Row>
                    <Col span={10}></Col>
                    <Col span={10}>
                      <Text className="mb-3 mt-1">
                        Actual model name used for making{" "}
                        <Link
                          href="https://docs.litellm.ai/docs/providers"
                          target="_blank"
                        >
                          litellm.completion() call
                        </Link>
                        . We&apos;ll{" "}
                        <Link
                          href="https://docs.litellm.ai/docs/proxy/reliability#step-1---set-deployments-on-config"
                          target="_blank"
                        >
                          loadbalance
                        </Link>{" "}
                        models with the same &apos;public name&apos;
                      </Text>
                    </Col>
                  </Row>
                  {dynamicProviderForm !== undefined &&
                    dynamicProviderForm.fields.length > 0 && (
                      <DynamicFields
                        fields={dynamicProviderForm.fields}
                        selectedProvider={dynamicProviderForm.name}
                      />
                    )}
                  {selectedProvider != Providers.Bedrock &&
                    selectedProvider != Providers.Vertex_AI &&
                    (dynamicProviderForm === undefined ||
                      dynamicProviderForm.fields.length == 0) && (
                      <Form.Item
                        rules={[{ required: true, message: "Required" }]}
                        label="API Key"
                        name="api_key"
                      >
                        <TextInput placeholder="sk-" type="password" />
                      </Form.Item>
                    )}
                  {selectedProvider == Providers.OpenAI && (
                    <Form.Item label="Organization ID" name="organization_id">
                      <TextInput placeholder="[OPTIONAL] my-unique-org" />
                    </Form.Item>
                  )}
                  {selectedProvider == Providers.Vertex_AI && (
                    <Form.Item
                      rules={[{ required: true, message: "Required" }]}
                      label="Vertex Project"
                      name="vertex_project"
                    >
                      <TextInput placeholder="adroit-cadet-1234.." />
                    </Form.Item>
                  )}
                  {selectedProvider == Providers.Vertex_AI && (
                    <Form.Item
                      rules={[{ required: true, message: "Required" }]}
                      label="Vertex Location"
                      name="vertex_location"
                    >
                      <TextInput placeholder="us-east-1" />
                    </Form.Item>
                  )}
                  {selectedProvider == Providers.Vertex_AI && (
                    <Form.Item
                      rules={[{ required: true, message: "Required" }]}
                      label="Vertex Credentials"
                      name="vertex_credentials"
                      className="mb-0"
                    >
                      <Upload {...props}>
                        <Button2 icon={<UploadOutlined />}>
                          Click to Upload
                        </Button2>
                      </Upload>
                    </Form.Item>
                  )}
                  {selectedProvider == Providers.Vertex_AI && (
                    <Row>
                      <Col span={10}></Col>
                      <Col span={10}>
                        <Text className="mb-3 mt-1">
                          Give litellm a gcp service account(.json file), so it
                          can make the relevant calls
                        </Text>
                      </Col>
                    </Row>
                  )}
                  {(selectedProvider == Providers.Azure ||
                    selectedProvider == Providers.OpenAI_Compatible) && (
                    <Form.Item
                      rules={[{ required: true, message: "Required" }]}
                      label="API Base"
                      name="api_base"
                    >
                      <TextInput placeholder="https://..." />
                    </Form.Item>
                  )}
                  {selectedProvider == Providers.Azure && (
                    <Form.Item
                      rules={[{ required: true, message: "Required" }]}
                      label="API Version"
                      name="api_version"
                    >
                      <TextInput placeholder="2023-07-01-preview" />
                    </Form.Item>
                  )}
                  {selectedProvider == Providers.Azure && (
                    <div>
                      <Form.Item
                        label="Base Model"
                        name="base_model"
                        className="mb-0"
                      >
                        <TextInput placeholder="azure/gpt-3.5-turbo" />
                      </Form.Item>
                      <Row>
                        <Col span={10}></Col>
                        <Col span={10}>
                          <Text className="mb-2">
                            The actual model your azure deployment uses. Used
                            for accurate cost tracking. Select name from{" "}
                            <Link
                              href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json"
                              target="_blank"
                            >
                              here
                            </Link>
                          </Text>
                        </Col>
                      </Row>
                    </div>
                  )}
                  {selectedProvider == Providers.Bedrock && (
                    <Form.Item
                      rules={[{ required: true, message: "Required" }]}
                      label="AWS Access Key ID"
                      name="aws_access_key_id"
                      tooltip="You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`)."
                    >
                      <TextInput placeholder="" />
                    </Form.Item>
                  )}
                  {selectedProvider == Providers.Bedrock && (
                    <Form.Item
                      rules={[{ required: true, message: "Required" }]}
                      label="AWS Secret Access Key"
                      name="aws_secret_access_key"
                      tooltip="You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`)."
                    >
                      <TextInput placeholder="" />
                    </Form.Item>
                  )}
                  {selectedProvider == Providers.Bedrock && (
                    <Form.Item
                      rules={[{ required: true, message: "Required" }]}
                      label="AWS Region Name"
                      name="aws_region_name"
                      tooltip="You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`)."
                    >
                      <TextInput placeholder="us-east-1" />
                    </Form.Item>
                  )}
                  <Form.Item
                    label="LiteLLM Params"
                    name="litellm_extra_params"
                    tooltip="Optional litellm params used for making a litellm.completion() call."
                    className="mb-0"
                  >
                    <TextArea
                      rows={4}
                      placeholder='{
                    "rpm": 100,
                    "timeout": 0,
                    "stream_timeout": 0
                  }'
                    />
                  </Form.Item>
                  <Row>
                    <Col span={10}></Col>
                    <Col span={10}>
                      <Text className="mb-3 mt-1">
                        Pass JSON of litellm supported params{" "}
                        <Link
                          href="https://docs.litellm.ai/docs/completion/input"
                          target="_blank"
                        >
                          litellm.completion() call
                        </Link>
                      </Text>
                    </Col>
                  </Row>
                </>
                <div style={{ textAlign: "center", marginTop: "10px" }}>
                  <Button2 htmlType="submit">Add Model</Button2>
                </div>
                <Tooltip title="Get help on our github">
                  <Typography.Link href="https://github.com/BerriAI/litellm/issues">
                    Need Help?
                  </Typography.Link>
                </Tooltip>
              </Form>
            </Card>
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
            {/* <p style={{fontSize: '0.85rem', color: '#808080'}}>View how requests were load balanced within a model group</p> */}

            <Grid numItems={2} className="mt-2">
              <Col>
                <Text>Select Time Range</Text>
                <DateRangePicker
                  enableSelect={true}
                  value={dateValue}
                  onValueChange={(value) => {
                    setDateValue(value);
                    updateModelMetrics(
                      selectedModelGroup,
                      value.from,
                      value.to
                    ); // Call updateModelMetrics with the new date range
                  }}
                />
              </Col>
              <Col>
                <Text>Select Model Group</Text>
                <Select
                  className="mb-4 mt-2"
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
                >
                  {availableModelGroups.map((group, idx) => (
                    <SelectItem
                      key={idx}
                      value={group}
                      onClick={() =>
                        updateModelMetrics(group, dateValue.from, dateValue.to)
                      }
                    >
                      {group}
                    </SelectItem>
                  ))}
                </Select>
              </Col>
            </Grid>

            <Grid numItems={2}>
              <Col>
                <Card className="mr-2 max-h-[400px] min-h-[400px]">
                  <TabGroup>
                    <TabList variant="line" defaultValue="1">
                      <Tab value="1">Avg. Latency per Token</Tab>
                      <Tab value="2">✨ Time to first token</Tab>
                    </TabList>
                    <TabPanels>
                      <TabPanel>
                        <p className="text-gray-500 italic"> (seconds/token)</p>
                        <Text className="text-gray-500 italic mt-1 mb-1">
                          average Latency for successfull requests divided by
                          the total tokens
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
                          modelMetricsCategories={
                            streamingModelMetricsCategories
                          }
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
                <Title>All Up Rate Limit Errors (429) for {selectedModelGroup}</Title>
                <Grid numItems={1}>
                <Col>
                <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452"}}>Num Rate Limit Errors { (globalExceptionData.sum_num_rate_limit_exceptions)}</Subtitle>
                <BarChart
                    className="h-40"
                    data={globalExceptionData.daily_data}
                    index="date"
                    colors={['rose']}
                    categories={['num_rate_limit_exceptions']}
                    onValueChange={(v) => console.log(v)}
                  />
                  </Col>
                  <Col>

                {/* <BarChart
                    className="h-40"
                    data={modelExceptions}
                    index="model"
                    categories={allExceptions}
                    stack={true}
                    yAxisWidth={30}
              /> */}
      

                </Col>

                </Grid>
                

                </Card>

                {
                  premiumUser ? ( 
                    <>
                    {globalExceptionPerDeployment.map((globalActivity, index) => (
                <Card key={index}>
                  <Title>{globalActivity.api_base ? globalActivity.api_base : "Unknown API Base"}</Title>
                  <Grid numItems={1}>
                    <Col>
                      <Subtitle style={{ fontSize: "15px", fontWeight: "normal", color: "#535452"}}>Num Rate Limit Errors (429) {(globalActivity.sum_num_rate_limit_exceptions)}</Subtitle>
                      <BarChart
                        className="h-40"
                        data={globalActivity.daily_data}
                        index="date"
                        colors={['rose']}
                        categories={['num_rate_limit_exceptions']}
          
                        onValueChange={(v) => console.log(v)}
                      />
                      
                    </Col>
                  </Grid>
                </Card>
              ))}
                    </>
                  ) : 
                  <>
                  {globalExceptionPerDeployment && globalExceptionPerDeployment.length > 0 &&
                    globalExceptionPerDeployment.slice(0, 1).map((globalActivity, index) => (
                      <Card key={index}>
                        <Title>✨ Rate Limit Errors by Deployment</Title>
                        <p className="mb-2 text-gray-500 italic text-[12px]">Upgrade to see exceptions for all deployments</p>
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
                              Num Rate Limit Errors {(globalActivity.sum_num_rate_limit_exceptions)}
                            </Subtitle>
                            <BarChart
                                className="h-40"
                                data={globalActivity.daily_data}
                                index="date"
                                colors={['rose']}
                                categories={['num_rate_limit_exceptions']}
                  
                                onValueChange={(v) => console.log(v)}
                              />
                          </Col>
                          
                          
                        </Grid>
                        </Card>
                      </Card>
                    ))}
                </>
                }              
              </Grid>
              
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
    </div>
  );
};

export default ModelDashboard;
