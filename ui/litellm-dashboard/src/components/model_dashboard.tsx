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
import { TabPanel, TabPanels, TabGroup, TabList, Tab, TextInput, Icon } from "@tremor/react";
import { Select, SelectItem, MultiSelect, MultiSelectItem } from "@tremor/react";
import { modelInfoCall, userGetRequesedtModelsCall, modelCreateCall, Model, modelCostMap, modelDeleteCall, healthCheckCall, modelUpdateCall } from "./networking";
import { BarChart } from "@tremor/react";
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
  Row, Col,
} from "antd";
import { Badge, BadgeDelta, Button } from "@tremor/react";
import RequestAccess from "./request_model_access";
import { Typography } from "antd";
import TextArea from "antd/es/input/TextArea";
import { InformationCircleIcon, PencilAltIcon, PencilIcon, StatusOnlineIcon, TrashIcon, RefreshIcon } from "@heroicons/react/outline";
import DeleteModelButton from "./delete_model_button";
const { Title: Title2, Link } = Typography;
import { UploadOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';
import { Upload } from 'antd';

interface ModelDashboardProps {
  accessToken: string | null;
  token: string | null;
  userRole: string | null;
  userID: string | null;
  modelData: any, 
  setModelData: any
}

interface EditModelModalProps {
  visible: boolean;
  onCancel: () => void;
  model: any; // Assuming TeamType is a type representing your team object
  onSubmit: (data: FormData) => void; // Assuming FormData is the type of data to be submitted
}

//["OpenAI", "Azure OpenAI", "Anthropic", "Gemini (Google AI Studio)", "Amazon Bedrock", "OpenAI-Compatible Endpoints (Groq, Together AI, Mistral AI, etc.)"]

enum Providers {
  OpenAI = "OpenAI",
  Azure = "Azure",
  Anthropic = "Anthropic",
  Google_AI_Studio = "Gemini (Google AI Studio)",
  Bedrock = "Amazon Bedrock",
  OpenAI_Compatible = "OpenAI-Compatible Endpoints (Groq, Together AI, Mistral AI, etc.)",
  Vertex_AI = "Vertex AI (Anthropic, Gemini, etc.)"
}

const provider_map: Record <string, string> = {
  "OpenAI": "openai",
  "Azure": "azure",
  "Anthropic": "anthropic",
  "Google_AI_Studio": "gemini",
  "Bedrock": "bedrock",
  "OpenAI_Compatible": "openai",
  "Vertex_AI": "vertex_ai"
};


const handleSubmit = async (formValues: Record<string, any>, accessToken: string, form: any) => {
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
    let deployments: Array<string> = Array.isArray(formValues["model"]) ? formValues["model"] : [formValues["model"]];
    console.log(`received deployments: ${deployments}`)
    console.log(`received type of deployments: ${typeof deployments}`)
    deployments.forEach(async (litellm_model) => { 
      console.log(`litellm_model: ${litellm_model}`)
      const litellmParamsObj: Record<string, any>  = {};
      const modelInfoObj: Record<string, any>  = {};
      // Iterate through the key-value pairs in formValues
      litellmParamsObj["model"] = litellm_model
      let modelName: string  = "";
      for (const [key, value] of Object.entries(formValues)) {
        if (value === '') {
          continue;
        }
        if (key == "model_name") {
          modelName = modelName + value
        }
        else if (key == "custom_llm_provider") {
          // const providerEnumValue = Providers[value as keyof typeof Providers];
          // const mappingResult = provider_map[providerEnumValue]; // Get the corresponding value from the mapping
          // modelName = mappingResult + "/" + modelName
          continue
        }
        else if (key == "model") {
          continue
        }

        // Check if key is "base_model"
        else if (key === "base_model") {
          // Add key-value pair to model_info dictionary
          modelInfoObj[key] = value;
        }

        else if (key == "litellm_extra_params") {
          console.log("litellm_extra_params:", value);
          let litellmExtraParams = {};
          if (value && value != undefined) {
            try {
              litellmExtraParams = JSON.parse(value);
            }
            catch (error) {
              message.error("Failed to parse LiteLLM Extra Params: " + error, 20);
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
        "model_name": modelName,
        "litellm_params": litellmParamsObj,
        "model_info": modelInfoObj
      }

      

      const response: any = await modelCreateCall(
        accessToken,
        new_model
      );

      console.log(`response for model create call: ${response["data"]}`);
    }); 
    
    form.resetFields();

    
    } catch (error) {
      message.error("Failed to create model: " + error, 20);
    }
}

const ModelDashboard: React.FC<ModelDashboardProps> = ({
  accessToken,
  token,
  userRole,
  userID,
  modelData = { data: [] },
  setModelData
}) => {
  const [pendingRequests, setPendingRequests] = useState<any[]>([]);
  const [form] = Form.useForm();
  const [modelMap, setModelMap] = useState<any>(null);
  const [lastRefreshed, setLastRefreshed] = useState('');

  const [providerModels, setProviderModels] = useState<Array<string>>([]); // Explicitly typing providerModels as a string array

  const providers = Object.values(Providers).filter(key => isNaN(Number(key)));

  
  const [selectedProvider, setSelectedProvider] = useState<String>("OpenAI");
  const [healthCheckResponse, setHealthCheckResponse] = useState<string>('');
  const [editModalVisible, setEditModalVisible] = useState<boolean>(false);
  const [selectedModel, setSelectedModel] = useState<any>(null);
  const [availableModelGroups, setAvailableModelGroups] = useState<Array<string>>([]);
  const [selectedModelGroup, setSelectedModelGroup] = useState<string | null>(null);

  const EditModelModal: React.FC<EditModelModalProps> = ({ visible, onCancel, model, onSubmit }) => {
    const [form] = Form.useForm();
    let litellm_params_to_edit: Record<string, any> = {}
    let model_name = "";
    let model_id = "";
    if (model) {
      litellm_params_to_edit = model.litellm_params
      model_name = model.model_name;
      let model_info = model.model_info;
      if (model_info ) {
        model_id = model_info.id;
        console.log(`model_id: ${model_id}`)
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

                <Form.Item className="mt-8"
                    label="api_base" 
                    name="api_base"
                    
                  >
                  <TextInput/>

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

                  <Form.Item 
                    label="max_retries" 
                    name="max_retries"
                  >

                    
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
                    label="model_id" 
                    name="model_id"
                    hidden={true}
                  >
                  </Form.Item>

                  

                  
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

  const handleEditCancel = () => {
    setEditModalVisible(false);
    setSelectedModel(null);
  };


const handleEditSubmit = async (formValues: Record<string, any>) => {
  // Call API to update team with teamId and values
  
  console.log("handleEditSubmit:", formValues);
  if (accessToken == null) {
    return;
  }

  let newLiteLLMParams: Record<string, any> = {}
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
      "id": model_info_model_id
    }
  }

  console.log("handleEditSubmit payload:", payload);

  let newModelValue = await modelUpdateCall(accessToken, payload);

  // Update the teams state with the updated team data
  // if (teams) {
  //   const updatedTeams = teams.map((team) =>
  //     team.team_id === teamId ? newTeamValues.data : team
  //   );
  //   setTeams(updatedTeams);
  // }
  message.success("Model updated successfully, restart server to see updates");

  setEditModalVisible(false);
  setSelectedModel(null);
};


  


  const props: UploadProps = {
    name: 'file',
    accept: '.json', 
    beforeUpload: file => {
      if (file.type === 'application/json') {
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
      if (info.file.status !== 'uploading') {
        console.log(info.file, info.fileList);
      }
      if (info.file.status === 'done') {
        message.success(`${info.file.name} file uploaded successfully`);
      } else if (info.file.status === 'error') {
        message.error(`${info.file.name} file upload failed.`);
      }
    },
  };

  const handleRefreshClick = () => {
    // Update the 'lastRefreshed' state to the current date and time
    const currentDate = new Date();
    setLastRefreshed(currentDate.toLocaleString());
  };


  useEffect(() => {
    if (!accessToken || !token || !userRole || !userID) {
      return;
    }
    const fetchData = async () => {
      try {
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
          all_model_groups.add(model.model_name)
        }
        console.log("all_model_groups:", all_model_groups)
        let _array_model_groups = Array.from(all_model_groups)
        setAvailableModelGroups(_array_model_groups);

        // if userRole is Admin, show the pending requests
        if (userRole === "Admin" && accessToken) {
          const user_requests = await userGetRequesedtModelsCall(accessToken);
          console.log("Pending Requests:", pendingRequests);
          setPendingRequests(user_requests.requests || []);
        }
      } catch (error) {
        console.error("There was an error fetching the model data", error);
      }
    };

    if (accessToken && token && userRole && userID) {
      fetchData();
    }

    const fetchModelMap = async () => {
      const data = await modelCostMap()
      console.log(`received model cost map data: ${Object.keys(data)}`)
      setModelMap(data)
    }
    if (modelMap == null) {
      fetchModelMap()
    }

    handleRefreshClick()
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
    let litellm_model_name = curr_model?.litellm_params?.model
    let model_info = curr_model?.model_info;

    let defaultProvider = "openai";
    let provider = "";
    let input_cost = "Undefined";
    let output_cost = "Undefined";
    let max_tokens = "Undefined";
    let cleanedLitellmParams = {};

    const getProviderFromModel = (model: string) => {
      /**
       * Use model map
       * - check if model in model map
       * - return it's litellm_provider, if so 
       */
      console.log(`GET PROVIDER CALLED! - ${modelMap}`)
      if (modelMap !== null && modelMap !== undefined) {
        if (typeof modelMap == "object" && model in modelMap) {
          return modelMap[model]["litellm_provider"]
        }
      }
      return "openai"
    }

    // Check if litellm_model_name is null or undefined
    if (litellm_model_name) {
      // Split litellm_model_name based on "/"
      let splitModel = litellm_model_name.split("/");

      // Get the first element in the split
      let firstElement = splitModel[0];

      // If there is only one element, default provider to openai
      provider = splitModel.length === 1 ? getProviderFromModel(litellm_model_name) : firstElement;
    } else {
      // litellm_model_name is null or undefined, default provider to openai
      provider = "openai"
    }

    if (model_info) {
      input_cost = model_info?.input_cost_per_token;
      output_cost = model_info?.output_cost_per_token;
      max_tokens = model_info?.max_tokens;
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
    modelData.data[i].max_tokens = max_tokens;
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
    console.log(`received provider string: ${provider}`)
    const providerKey = Object.keys(Providers).find(key => (Providers as {[index: string]: any})[key] === provider);

    if (providerKey) {
      const mappingResult = provider_map[providerKey]; // Get the corresponding value from the mapping
      console.log(`mappingResult: ${mappingResult}`)
      let _providerModels: Array<string> = []
      if (typeof modelMap === 'object') {
        Object.entries(modelMap).forEach(([key, value]) => {
          if (
            value !== null 
            && typeof value === 'object'
            && "litellm_provider" in (value as object)
            && (
              (value as any)["litellm_provider"] === mappingResult 
              || (value as any)["litellm_provider"].includes(mappingResult)
            )) {
            _providerModels.push(key);
          }
        });
      }
      setProviderModels(_providerModels)
      console.log(`providerModels: ${providerModels}`);
    }
  }

  const runHealthCheck = async () => {
    try {
      message.info('Running health check...');
      setHealthCheckResponse('');
      const response = await healthCheckCall(accessToken);
      setHealthCheckResponse(response);
    } catch (error) {
      console.error('Error running health check:', error);
      setHealthCheckResponse('Error running health check');
    }
  };



  const getPlaceholder = (selectedProvider: string): string => {
    if (selectedProvider === Providers.Vertex_AI) {
      return 'gemini-pro';
    } else if (selectedProvider == Providers.Anthropic) {
      return 'claude-3-opus'
    } else if (selectedProvider == Providers.Bedrock) {
      return 'claude-3-opus'
    } else if (selectedProvider == Providers.Google_AI_Studio) {
      return 'gemini-pro'
    } else {
      return 'gpt-3.5-turbo';
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

  console.log(`selectedProvider: ${selectedProvider}`)
  console.log(`providerModels.length: ${providerModels.length}`)
  return (
    <div style={{ width: "100%", height: "100%"}}>
      <TabGroup className="gap-2 p-8 h-[75vh] w-full mt-2">
      <TabList className="flex justify-between mt-2 w-full items-center">
        <div className="flex">
          <Tab>All Models</Tab>
          <Tab>Add Model</Tab>
          <Tab><pre>/health Models</pre></Tab>
        </div>

        <div className="flex items-center space-x-2">
          {lastRefreshed && (
            <Text>
              Last Refreshed: {lastRefreshed}
            </Text>
          )}
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
              defaultValue="all"
              onValueChange={(value) => setSelectedModelGroup(value === "all" ? "all" : value)}
            >
              <SelectItem 
                  value={"all"}
                >
                  All Models
                </SelectItem>
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
          <Table className="mt-5">
            <TableHead>
              <TableRow>

                  <TableHeaderCell>Public Model Name </TableHeaderCell>

                <TableHeaderCell>
                  Provider
                </TableHeaderCell>
                {
                  userRole === "Admin" && (
                    <TableHeaderCell>
                      API Base
                    </TableHeaderCell>
                  )
                }
                <TableHeaderCell>
                  Extra litellm Params
                </TableHeaderCell>
                <TableHeaderCell>Input Price per token ($)</TableHeaderCell>
                <TableHeaderCell>Output Price per token ($)</TableHeaderCell>
                <TableHeaderCell>Max Tokens</TableHeaderCell>
              </TableRow>
            </TableHead>
            <TableBody>
              { modelData.data
                  .filter((model: any) =>
                    selectedModelGroup === "all" || model.model_name === selectedModelGroup || selectedModelGroup === null || selectedModelGroup === undefined || selectedModelGroup === ""
                  )
                  .map((model: any, index: number) => (
                <TableRow key={index}>
                  <TableCell>
                    <Text>{model.model_name}</Text>
                  </TableCell>
                  <TableCell>{model.provider}</TableCell>
                  {
                    userRole === "Admin" && (
                      <TableCell>{model.api_base}</TableCell>
                    )
                  }

                  <TableCell>

                <Accordion>
                  <AccordionHeader>
                    <Text>Litellm params</Text>
                  </AccordionHeader>
                  <AccordionBody>
                  <pre>
                    {JSON.stringify(model.cleanedLitellmParams, null, 2)}
                    </pre>
                  </AccordionBody>
                </Accordion>
                   
                  </TableCell>

                  <TableCell>{model.input_cost}</TableCell>
                  <TableCell>{model.output_cost}</TableCell>
                  <TableCell>{model.max_tokens}</TableCell>
                  <TableCell>
                        <Icon
                            icon={PencilAltIcon}
                            size="sm"
                            onClick={() => handleEditClick(model)}
                          />
                          <DeleteModelButton modelID={model.model_info.id} accessToken={accessToken} />
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
              <Form.Item rules={[{ required: true, message: 'Required' }]} label="Provider:" name="custom_llm_provider" tooltip="E.g. OpenAI, Azure OpenAI, Anthropic, Bedrock, etc." labelCol={{ span: 10 }} labelAlign="left">
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
              <Form.Item rules={[{ required: true, message: 'Required' }]} label="Public Model Name" name="model_name" tooltip="Model name your users will pass in. Also used for load-balancing, LiteLLM will load balance between all models with this public name." className="mb-0">
                  <TextInput placeholder={getPlaceholder(selectedProvider.toString())}/>
                </Form.Item>
                <Row>
                <Col span={10}></Col>
                <Col span={10}><Text className="mb-3 mt-1">Model name your users will pass in.</Text></Col>
                </Row>
                <Form.Item rules={[{ required: true, message: 'Required' }]} label="LiteLLM Model Name(s)" name="model" tooltip="Actual model name used for making litellm.completion() call." className="mb-0">
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
                <Col span={10}><Text className="mb-3 mt-1">Actual model name used for making <Link href="https://docs.litellm.ai/docs/providers" target="_blank">litellm.completion() call</Link>. We&apos;ll <Link href="https://docs.litellm.ai/docs/proxy/reliability#step-1---set-deployments-on-config" target="_blank">loadbalance</Link> models with the same &apos;public name&apos;</Text></Col></Row>
                {
                  selectedProvider != Providers.Bedrock && selectedProvider != Providers.Vertex_AI && <Form.Item
                  rules={[{ required: true, message: 'Required' }]}
                    label="API Key"
                    name="api_key"
                  >
                    <TextInput placeholder="sk-" type="password"/>
                  </Form.Item>
                }
                {
                  selectedProvider == Providers.OpenAI && <Form.Item
                    label="Organization ID"
                    name="organization_id"
                  >
                    <TextInput placeholder="[OPTIONAL] my-unique-org"/>
                  </Form.Item>
                }
                {
                  selectedProvider == Providers.Vertex_AI && <Form.Item rules={[{ required: true, message: 'Required' }]}
                  label="Vertex Project"
                  name="vertex_project"><TextInput placeholder="adroit-cadet-1234.."/></Form.Item>
                }
                {
                  selectedProvider == Providers.Vertex_AI && <Form.Item rules={[{ required: true, message: 'Required' }]}
                  label="Vertex Location"
                  name="vertex_location"><TextInput placeholder="us-east-1"/></Form.Item>
                }
                {
                  selectedProvider == Providers.Vertex_AI && <Form.Item rules={[{ required: true, message: 'Required' }]}
                  label="Vertex Credentials"
                  name="vertex_credentials"
                  className="mb-0">
                    <Upload {...props}>
                      <Button2 icon={<UploadOutlined />}>Click to Upload</Button2>
                    </Upload>
                  </Form.Item>
                }
                {
                  selectedProvider == Providers.Vertex_AI && <Row>
                  <Col span={10}></Col>
                  <Col span={10}><Text className="mb-3 mt-1">Give litellm a gcp service account(.json file), so it can make the relevant calls</Text></Col></Row>
  
                }
                {
                  (selectedProvider == Providers.Azure || selectedProvider == Providers.OpenAI_Compatible) && <Form.Item
                  rules={[{ required: true, message: 'Required' }]}
                  label="API Base"
                  name="api_base"
                >
                  <TextInput placeholder="https://..."/>
                </Form.Item>
                }
                {
                  selectedProvider == Providers.Azure && <Form.Item
                  rules={[{ required: true, message: 'Required' }]}
                  label="API Version"
                  name="api_version"
                >
                  <TextInput placeholder="2023-07-01-preview"/>
                </Form.Item>
                }
                {
                  selectedProvider == Providers.Azure && <Form.Item
                  label="Base Model"
                  name="base_model"
                >
                  <TextInput placeholder="azure/gpt-3.5-turbo"/>
                  <Text>The actual model your azure deployment uses. Used for accurate cost tracking. Select name from <Link href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json" target="_blank">here</Link></Text>
                </Form.Item>
                }
                {
                  selectedProvider == Providers.Bedrock && <Form.Item
                  rules={[{ required: true, message: 'Required' }]}
                  label="AWS Access Key ID"
                  name="aws_access_key_id"
                  tooltip="You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`)."
                >
                  <TextInput placeholder=""/>
                </Form.Item>
                }
                {
                  selectedProvider == Providers.Bedrock && <Form.Item
                  rules={[{ required: true, message: 'Required' }]}
                  label="AWS Secret Access Key"
                  name="aws_secret_access_key"
                  tooltip="You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`)."
                >
                  <TextInput placeholder=""/>
                </Form.Item>
                }
                {
                  selectedProvider == Providers.Bedrock && <Form.Item
                  rules={[{ required: true, message: 'Required' }]}
                  label="AWS Region Name"
                  name="aws_region_name"
                  tooltip="You can provide the raw key or the environment variable (e.g. `os.environ/MY_SECRET_KEY`)."
                >
                  <TextInput placeholder="us-east-1"/>
                </Form.Item>
                }
                <Form.Item label="LiteLLM Params" name="litellm_extra_params" tooltip="Optional litellm params used for making a litellm.completion() call." className="mb-0">
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
                <Col span={10}><Text className="mb-3 mt-1">Pass JSON of litellm supported params <Link href="https://docs.litellm.ai/docs/completion/input" target="_blank">litellm.completion() call</Link></Text></Col>
                </Row>

              </>
              <div style={{ textAlign: "center", marginTop: "10px" }}>
                <Button2 htmlType="submit">Add Model</Button2>
              </div>
              <Tooltip title="Get help on our github">
                <Typography.Link href="https://github.com/BerriAI/litellm/issues">Need Help?</Typography.Link>
              </Tooltip>
        </Form>
      </Card>
      </TabPanel>
      <TabPanel>
        <Card>
          <Text>`/health` will run a very small request through your models configured on litellm</Text>

          <Button onClick={runHealthCheck}>Run `/health`</Button>
          {healthCheckResponse && (
                <pre>{JSON.stringify(healthCheckResponse, null, 2)}</pre>
              )}

        </Card>
      </TabPanel>
      </TabPanels>
      </TabGroup>
      
    </div>
  );
};

export default ModelDashboard;