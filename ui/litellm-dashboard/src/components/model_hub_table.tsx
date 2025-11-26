import { CopyOutlined } from "@ant-design/icons";
import { Table as TableInstance } from "@tanstack/react-table";
import { Badge, Button, Card, Tab, TabGroup, TabList, TabPanel, TabPanels, Text, Title } from "@tremor/react";
import { Modal } from "antd";
import { Copy } from "lucide-react";
import { useRouter } from "next/navigation";
import React, { useCallback, useEffect, useRef, useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { isAdminRole } from "../utils/roles";
import { agentHubColumns, AgentHubData } from "./agent_hub_table_columns";
import MakeAgentPublicForm from "./make_agent_public_form";
import MakeMCPPublicForm from "./make_mcp_public_form";
import MakeModelPublicForm from "./make_model_public_form";
import { mcpHubColumns, MCPServerData } from "./mcp_hub_table_columns";
import { ModelDataTable } from "./model_dashboard/table";
import ModelFilters from "./model_filters";
import { modelHubColumns } from "./model_hub_table_columns";
import NotificationsManager from "./molecules/notifications_manager";
import {
  fetchMCPServers,
  getAgentsList,
  getConfigFieldSetting,
  getProxyBaseUrl,
  getUiConfig,
  modelHubCall,
  modelHubPublicModelsCall,
} from "./networking";
import PublicModelHub from "./public_model_hub";
import UsefulLinksManagement from "./useful_links_management";

interface ModelHubTableProps {
  accessToken: string | null;
  publicPage: boolean;
  premiumUser: boolean;
  userRole: string | null;
}

interface ModelGroupInfo {
  model_group: string;
  providers: string[];
  max_input_tokens?: number;
  max_output_tokens?: number;
  input_cost_per_token?: number;
  output_cost_per_token?: number;
  mode?: string;
  tpm?: number;
  rpm?: number;
  supports_parallel_function_calling: boolean;
  supports_vision: boolean;
  supports_function_calling: boolean;
  supported_openai_params?: string[];
  is_public_model_group: boolean;
  // Allow any additional properties for flexibility
  [key: string]: any;
}

const ModelHubTable: React.FC<ModelHubTableProps> = ({ accessToken, publicPage, premiumUser, userRole }) => {
  const [publicPageAllowed, setPublicPageAllowed] = useState<boolean>(false);
  const [modelHubData, setModelHubData] = useState<ModelGroupInfo[] | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [isPublicPageModalVisible, setIsPublicPageModalVisible] = useState(false);
  const [selectedModel, setSelectedModel] = useState<null | ModelGroupInfo>(null);
  const [filteredData, setFilteredData] = useState<ModelGroupInfo[]>([]);
  const [isMakePublicModalVisible, setIsMakePublicModalVisible] = useState(false);
  // Agent Hub state
  const [agentHubData, setAgentHubData] = useState<AgentHubData[] | null>(null);
  const [isMakeAgentPublicModalVisible, setIsMakeAgentPublicModalVisible] = useState(false);
  const [agentLoading, setAgentLoading] = useState<boolean>(true);
  const [selectedAgent, setSelectedAgent] = useState<null | AgentHubData>(null);
  const [isAgentModalVisible, setIsAgentModalVisible] = useState(false);
  // MCP Hub state
  const [mcpHubData, setMcpHubData] = useState<MCPServerData[] | null>(null);
  const [mcpLoading, setMcpLoading] = useState<boolean>(true);
  const [selectedMcpServer, setSelectedMcpServer] = useState<null | MCPServerData>(null);
  const [isMcpModalVisible, setIsMcpModalVisible] = useState(false);
  const [isMakeMcpPublicModalVisible, setIsMakeMcpPublicModalVisible] = useState(false);
  const router = useRouter();
  const tableRef = useRef<TableInstance<any>>(null);
  const agentTableRef = useRef<TableInstance<any>>(null);
  const mcpTableRef = useRef<TableInstance<any>>(null);

  useEffect(() => {
    const fetchData = async (accessToken: string) => {
      try {
        setLoading(true);
        const _modelHubData = await modelHubCall(accessToken);
        console.log("ModelHubData:", _modelHubData);
        setModelHubData(_modelHubData.data);

        getConfigFieldSetting(accessToken, "enable_public_model_hub")
          .then((data) => {
            console.log(`data: ${JSON.stringify(data)}`);
            if (data.field_value == true) {
              setPublicPageAllowed(true);
            }
          })
          .catch((error) => {
            // do nothing
          });
      } catch (error) {
        console.error("There was an error fetching the model data", error);
      } finally {
        setLoading(false);
      }
    };

    const fetchPublicData = async () => {
      try {
        setLoading(true);
        await getUiConfig();
        const _modelHubData = await modelHubPublicModelsCall();
        console.log("ModelHubData:", _modelHubData);
        console.log("First model structure:", _modelHubData[0]);
        console.log("Model has model_group?", _modelHubData[0]?.model_group);
        console.log("Model has providers?", _modelHubData[0]?.providers);
        setModelHubData(_modelHubData);
        setPublicPageAllowed(true);
      } catch (error) {
        console.error("There was an error fetching the public model data", error);
      } finally {
        setLoading(false);
      }
    };

    if (accessToken) {
      fetchData(accessToken);
    } else if (publicPage) {
      fetchPublicData();
    }
  }, [accessToken, publicPage]);

  // Fetch Agent Hub data
  useEffect(() => {
    const fetchAgentData = async () => {
      if (!accessToken) {
        return;
      }

      try {
        setAgentLoading(true);
        const response = await getAgentsList(accessToken);
        console.log("AgentHubData:", response);
        let agents = response.agents;
        let agent_card_list = agents.map((agent: any) => ({
          agent_id: agent.agent_id,
          ...agent.agent_card_params,
          is_public: agent.litellm_params.is_public,
        }));
        setAgentHubData(agent_card_list);
      } catch (error) {
        console.error("There was an error fetching the agent data", error);
      } finally {
        setAgentLoading(false);
      }
    };

    if (!publicPage) {
      fetchAgentData();
    }
  }, [publicPage, accessToken]);

  // Fetch MCP Hub data
  useEffect(() => {
    const fetchMcpData = async () => {
      if (!accessToken) {
        return;
      }

      try {
        setMcpLoading(true);
        const response = await fetchMCPServers(accessToken);
        console.log("MCPHubData:", response);
        setMcpHubData(response);
      } catch (error) {
        console.error("There was an error fetching the MCP server data", error);
      } finally {
        setMcpLoading(false);
      }
    };

    if (!publicPage) {
      fetchMcpData();
    }
  }, [publicPage, accessToken]);

  const showModal = (model: ModelGroupInfo) => {
    setSelectedModel(model);
    setIsModalVisible(true);
  };

  const showAgentModal = (agent: AgentHubData) => {
    setSelectedAgent(agent);
    setIsAgentModalVisible(true);
  };

  const showMcpModal = (server: MCPServerData) => {
    setSelectedMcpServer(server);
    setIsMcpModalVisible(true);
  };

  const goToPublicModelPage = () => {
    router.replace(`/model_hub_table?key=${accessToken}`);
  };

  const handleMakePublicPage = () => {
    if (!accessToken) {
      return;
    }

    // Show the modal for selecting models to make public
    setIsMakePublicModalVisible(true);
  };

  const handleMakeAgentPublicPage = () => {
    if (!accessToken) {
      return;
    }

    // Show the modal for selecting agents to make public
    setIsMakeAgentPublicModalVisible(true);
  };

  const handleMakeMcpPublicPage = () => {
    if (!accessToken) {
      return;
    }

    // Show the modal for selecting MCP servers to make public
    setIsMakeMcpPublicModalVisible(true);
  };

  const handleOk = () => {
    setIsModalVisible(false);
    setIsPublicPageModalVisible(false);
    setSelectedModel(null);
    setIsAgentModalVisible(false);
    setSelectedAgent(null);
    setIsMcpModalVisible(false);
    setSelectedMcpServer(null);
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    setIsPublicPageModalVisible(false);
    setSelectedModel(null);
    setIsAgentModalVisible(false);
    setSelectedAgent(null);
    setIsMcpModalVisible(false);
    setSelectedMcpServer(null);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    NotificationsManager.success("Copied to clipboard!");
  };

  const formatCapabilityName = (key: string) => {
    // Remove 'supports_' prefix and convert snake_case to Title Case
    return key
      .replace(/^supports_/, "")
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  const getModelCapabilities = (model: ModelGroupInfo) => {
    // Find all properties that start with 'supports_' and are true
    return Object.entries(model)
      .filter(([key, value]) => key.startsWith("supports_") && value === true)
      .map(([key]) => key);
  };

  const formatCost = (cost: number) => {
    return `$${(cost * 1_000_000).toFixed(2)}`;
  };

  const handleMakePublicSuccess = () => {
    // Refresh the model hub data after successful public operation
    if (accessToken) {
      const fetchData = async () => {
        try {
          const _modelHubData = await modelHubCall(accessToken);
          setModelHubData(_modelHubData.data);
        } catch (error) {
          console.error("Error refreshing model data:", error);
        }
      };
      fetchData();
    }
  };

  const handleMakeAgentPublicSuccess = () => {
    // Refresh the agent hub data after successful public operation
    if (accessToken) {
      const fetchAgentData = async () => {
        try {
          const response = await getAgentsList(accessToken);
          let agents = response.agents;
          let agent_card_list = agents.map((agent: any) => ({
            agent_id: agent.agent_id,
            ...agent.agent_card_params,
            is_public: agent.is_public,
          }));
          setAgentHubData(agent_card_list);
        } catch (error) {
          console.error("Error refreshing agent data:", error);
        }
      };
      fetchAgentData();
    }
  };

  const handleMakeMcpPublicSuccess = () => {
    // Refresh the MCP hub data after successful public operation
    if (accessToken) {
      const fetchMcpData = async () => {
        try {
          const response = await fetchMCPServers(accessToken);
          setMcpHubData(response);
        } catch (error) {
          console.error("Error refreshing MCP server data:", error);
        }
      };
      fetchMcpData();
    }
  };

  const handleFilteredDataChange = useCallback((newFilteredData: ModelGroupInfo[]) => {
    setFilteredData(newFilteredData);
  }, []);

  console.log("publicPage: ", publicPage);
  console.log("publicPageAllowed: ", publicPageAllowed);

  // If this is a public page, use the dedicated PublicModelHub component
  if (publicPage && publicPageAllowed) {
    return <PublicModelHub accessToken={accessToken} />;
  }

  return (
    <div className="w-full mx-4 h-[75vh]">
      {publicPage == false ? (
        <div className="w-full m-2 mt-2 p-8">
          {/* Header with Title, Description and URL */}
          <div className="flex justify-between items-center mb-6">
            <div className="flex flex-col items-start">
              <Title className="text-center">AI Hub</Title>
              {isAdminRole(userRole || "") ? (
                <p className="text-sm text-gray-600">
                  Make models, agents, and MCP servers public for developers to know what&apos;s available.
                </p>
              ) : (
                <p className="text-sm text-gray-600">A list of all public model names personally available to you.</p>
              )}
            </div>
            <div className="flex items-center space-x-4">
              <Text>Model Hub URL:</Text>
              <div className="flex items-center bg-gray-200 px-2 py-1 rounded">
                <Text className="mr-2">{`${getProxyBaseUrl()}/ui/model_hub_table`}</Text>
                <button
                  onClick={() => copyToClipboard(`${getProxyBaseUrl()}/ui/model_hub_table`)}
                  className="p-1 hover:bg-gray-300 rounded transition-colors"
                  title="Copy URL"
                >
                  <Copy size={16} className="text-gray-600" />
                </button>
              </div>
            </div>
          </div>

          {/* Useful Links Management Section for Admins */}
          {isAdminRole(userRole || "") && (
            <div className="mt-8 mb-2">
              <UsefulLinksManagement accessToken={accessToken} userRole={userRole} />
            </div>
          )}

          {/* Tab System for Model Hub, Agent Hub, and MCP Hub */}
          <TabGroup>
            <TabList className="mb-4">
              <Tab>Model Hub</Tab>
              <Tab>Agent Hub</Tab>
              <Tab>MCP Hub</Tab>
            </TabList>

            <TabPanels>
              {/* Model Hub Tab */}
              <TabPanel>
                {/* Model Filters and Table */}
                <Card>
                  {/* Header with Make Public Button */}
                  {publicPage == false && isAdminRole(userRole || "") && (
                    <div className="flex justify-end mb-4">
                      <Button onClick={() => handleMakePublicPage()}>Select Models to Make Public</Button>
                    </div>
                  )}

                  {/* Filters */}
                  <ModelFilters modelHubData={modelHubData || []} onFilteredDataChange={handleFilteredDataChange} />

                  {/* Model Table */}
                  <ModelDataTable
                    columns={modelHubColumns(showModal, copyToClipboard, publicPage)}
                    data={filteredData}
                    isLoading={loading}
                    table={tableRef}
                    defaultSorting={[{ id: "model_group", desc: false }]}
                  />
                </Card>

                <div className="mt-4 text-center space-y-2">
                  <Text className="text-sm text-gray-600">
                    Showing {filteredData.length} of {modelHubData?.length || 0} models
                  </Text>
                </div>
              </TabPanel>

              {/* Agent Hub Tab */}
              <TabPanel>
                <Card>
                  {/* Header with Make Public Button */}
                  {publicPage == false && isAdminRole(userRole || "") && (
                    <div className="flex justify-end mb-4">
                      <Button onClick={() => handleMakeAgentPublicPage()}>Select Agents to Make Public</Button>
                    </div>
                  )}

                  {/* Agent Table */}
                  <ModelDataTable
                    columns={agentHubColumns(showAgentModal, copyToClipboard, publicPage)}
                    data={agentHubData || []}
                    isLoading={agentLoading}
                    table={agentTableRef}
                    defaultSorting={[{ id: "name", desc: false }]}
                  />
                </Card>

                <div className="mt-4 text-center space-y-2">
                  <Text className="text-sm text-gray-600">
                    Showing {agentHubData?.length || 0} agent{agentHubData?.length !== 1 ? "s" : ""}
                  </Text>
                </div>
              </TabPanel>

              {/* MCP Hub Tab */}
              <TabPanel>
                <Card>
                  {/* Header with Make Public Button */}
                  {publicPage == false && isAdminRole(userRole || "") && (
                    <div className="flex justify-end mb-4">
                      <Button onClick={() => handleMakeMcpPublicPage()}>Select MCP Servers to Make Public</Button>
                    </div>
                  )}

                  {/* MCP Server Table */}
                  <ModelDataTable
                    columns={mcpHubColumns(showMcpModal, copyToClipboard, publicPage)}
                    data={mcpHubData || []}
                    isLoading={mcpLoading}
                    table={mcpTableRef}
                    defaultSorting={[{ id: "server_name", desc: false }]}
                  />
                </Card>

                <div className="mt-4 text-center space-y-2">
                  <Text className="text-sm text-gray-600">
                    Showing {mcpHubData?.length || 0} MCP server{mcpHubData?.length !== 1 ? "s" : ""}
                  </Text>
                </div>
              </TabPanel>
            </TabPanels>
          </TabGroup>
        </div>
      ) : (
        <Card className="mx-auto max-w-xl mt-10">
          <Text className="text-xl text-center mb-2 text-black">Public Model Hub not enabled.</Text>
          <p className="text-base text-center text-slate-800">Ask your proxy admin to enable this on their Admin UI.</p>
        </Card>
      )}

      {/* Public Page Modal */}
      <Modal
        title="Public Model Hub"
        width={600}
        visible={isPublicPageModalVisible}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <div className="pt-5 pb-5">
          <div className="flex justify-between mb-4">
            <Text className="text-base mr-2">Shareable Link:</Text>
            <Text className="max-w-sm ml-2 bg-gray-200 pr-2 pl-2 pt-1 pb-1 text-center rounded">
              {`${getProxyBaseUrl()}/ui/model_hub_table`}
            </Text>
          </div>
          <div className="flex justify-end">
            <Button onClick={goToPublicModelPage}>See Page</Button>
          </div>
        </div>
      </Modal>

      {/* Model Details Modal */}
      <Modal
        title={selectedModel?.model_group || "Model Details"}
        width={1000}
        visible={isModalVisible}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        {selectedModel && (
          <div className="space-y-6">
            {/* Model Overview */}
            <div>
              <Text className="text-lg font-semibold mb-4">Model Overview</Text>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <Text className="font-medium">Model Group:</Text>
                  <Text>{selectedModel.model_group}</Text>
                </div>
                <div>
                  <Text className="font-medium">Mode:</Text>
                  <Text>{selectedModel.mode || "Not specified"}</Text>
                </div>
                <div>
                  <Text className="font-medium">Providers:</Text>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {selectedModel.providers.map((provider) => (
                      <Badge key={provider} color="blue">
                        {provider}
                      </Badge>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Token and Cost Information */}
            <div>
              <Text className="text-lg font-semibold mb-4">Token & Cost Information</Text>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Text className="font-medium">Max Input Tokens:</Text>
                  <Text>{selectedModel.max_input_tokens?.toLocaleString() || "Not specified"}</Text>
                </div>
                <div>
                  <Text className="font-medium">Max Output Tokens:</Text>
                  <Text>{selectedModel.max_output_tokens?.toLocaleString() || "Not specified"}</Text>
                </div>
                <div>
                  <Text className="font-medium">Input Cost per 1M Tokens:</Text>
                  <Text>
                    {selectedModel.input_cost_per_token
                      ? formatCost(selectedModel.input_cost_per_token)
                      : "Not specified"}
                  </Text>
                </div>
                <div>
                  <Text className="font-medium">Output Cost per 1M Tokens:</Text>
                  <Text>
                    {selectedModel.output_cost_per_token
                      ? formatCost(selectedModel.output_cost_per_token)
                      : "Not specified"}
                  </Text>
                </div>
              </div>
            </div>

            {/* Capabilities */}
            <div>
              <Text className="text-lg font-semibold mb-4">Capabilities</Text>
              <div className="flex flex-wrap gap-2">
                {(() => {
                  const capabilities = getModelCapabilities(selectedModel);
                  const colors = ["green", "blue", "purple", "orange", "red", "yellow"];

                  if (capabilities.length === 0) {
                    return <Text className="text-gray-500">No special capabilities listed</Text>;
                  }

                  return capabilities.map((capability, index) => (
                    <Badge key={capability} color={colors[index % colors.length]}>
                      {formatCapabilityName(capability)}
                    </Badge>
                  ));
                })()}
              </div>
            </div>

            {/* Rate Limits */}
            {(selectedModel.tpm || selectedModel.rpm) && (
              <div>
                <Text className="text-lg font-semibold mb-4">Rate Limits</Text>
                <div className="grid grid-cols-2 gap-4">
                  {selectedModel.tpm && (
                    <div>
                      <Text className="font-medium">Tokens per Minute:</Text>
                      <Text>{selectedModel.tpm.toLocaleString()}</Text>
                    </div>
                  )}
                  {selectedModel.rpm && (
                    <div>
                      <Text className="font-medium">Requests per Minute:</Text>
                      <Text>{selectedModel.rpm.toLocaleString()}</Text>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Supported OpenAI Parameters */}
            {selectedModel.supported_openai_params && (
              <div>
                <Text className="text-lg font-semibold mb-4">Supported OpenAI Parameters</Text>
                <div className="flex flex-wrap gap-2">
                  {selectedModel.supported_openai_params.map((param) => (
                    <Badge key={param} color="green">
                      {param}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Usage Example */}
            <div>
              <Text className="text-lg font-semibold mb-4">Usage Example</Text>
              <SyntaxHighlighter language="python" className="text-sm">
                {`import openai

client = openai.OpenAI(
    api_key="your_api_key",
    base_url="http://0.0.0.0:4000"  # Your LiteLLM Proxy URL
)

response = client.chat.completions.create(
    model="${selectedModel.model_group}",
    messages=[
        {
            "role": "user",
            "content": "Hello, how are you?"
        }
    ]
)

print(response.choices[0].message.content)`}
              </SyntaxHighlighter>
            </div>
          </div>
        )}
      </Modal>

      {/* Agent Details Modal */}
      <Modal
        title={selectedAgent?.name || "Agent Details"}
        width={1000}
        visible={isAgentModalVisible}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        {selectedAgent && (
          <div className="space-y-6">
            {/* Agent Overview */}
            <div>
              <Text className="text-lg font-semibold mb-4">Agent Overview</Text>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <Text className="font-medium">Name:</Text>
                  <Text>{selectedAgent.name}</Text>
                </div>
                <div>
                  <Text className="font-medium">Version:</Text>
                  <Badge color="blue">v{selectedAgent.version}</Badge>
                </div>
                <div>
                  <Text className="font-medium">Protocol Version:</Text>
                  <Text>{selectedAgent.protocolVersion}</Text>
                </div>
                <div>
                  <Text className="font-medium">URL:</Text>
                  <div className="flex items-center space-x-2">
                    <Text className="truncate">{selectedAgent.url}</Text>
                    <CopyOutlined
                      onClick={() => copyToClipboard(selectedAgent.url)}
                      className="cursor-pointer text-gray-500 hover:text-blue-500"
                    />
                  </div>
                </div>
              </div>
              <div>
                <Text className="font-medium">Description:</Text>
                <Text className="mt-1">{selectedAgent.description}</Text>
              </div>
            </div>

            {/* Capabilities */}
            {selectedAgent.capabilities && Object.keys(selectedAgent.capabilities).length > 0 && (
              <div>
                <Text className="text-lg font-semibold mb-4">Capabilities</Text>
                <div className="flex flex-wrap gap-2">
                  {Object.entries(selectedAgent.capabilities)
                    .filter(([_, value]) => value === true)
                    .map(([key]) => (
                      <Badge key={key} color="green">
                        {key}
                      </Badge>
                    ))}
                </div>
              </div>
            )}

            {/* Input/Output Modes */}
            <div>
              <Text className="text-lg font-semibold mb-4">Input/Output Modes</Text>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Text className="font-medium">Input Modes:</Text>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {selectedAgent.defaultInputModes?.map((mode) => (
                      <Badge key={mode} color="blue">
                        {mode}
                      </Badge>
                    )) || <Text>Not specified</Text>}
                  </div>
                </div>
                <div>
                  <Text className="font-medium">Output Modes:</Text>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {selectedAgent.defaultOutputModes?.map((mode) => (
                      <Badge key={mode} color="purple">
                        {mode}
                      </Badge>
                    )) || <Text>Not specified</Text>}
                  </div>
                </div>
              </div>
            </div>

            {/* Skills */}
            {selectedAgent.skills && selectedAgent.skills.length > 0 && (
              <div>
                <Text className="text-lg font-semibold mb-4">Skills</Text>
                <div className="space-y-4">
                  {selectedAgent.skills.map((skill) => (
                    <div key={skill.id} className="border border-gray-200 rounded p-4">
                      <div className="flex justify-between items-start mb-2">
                        <div>
                          <Text className="font-medium text-base">{skill.name}</Text>
                          <Text className="text-xs text-gray-500">ID: {skill.id}</Text>
                        </div>
                        {skill.tags && skill.tags.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {skill.tags.map((tag) => (
                              <Badge key={tag} color="purple" size="xs">
                                {tag}
                              </Badge>
                            ))}
                          </div>
                        )}
                      </div>
                      <Text className="text-sm mb-2">{skill.description}</Text>
                      {skill.examples && skill.examples.length > 0 && (
                        <div>
                          <Text className="text-xs font-medium text-gray-700">Examples:</Text>
                          <div className="flex flex-wrap gap-1 mt-1">
                            {skill.examples.map((example, idx) => (
                              <Badge key={idx} color="gray" size="xs">
                                {example}
                              </Badge>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Additional Properties */}
            {selectedAgent.supportsAuthenticatedExtendedCard && (
              <div>
                <Text className="text-lg font-semibold mb-4">Additional Features</Text>
                <Badge color="green">Supports Authenticated Extended Card</Badge>
              </div>
            )}
          </div>
        )}
      </Modal>

      {/* MCP Server Details Modal */}
      <Modal
        title={selectedMcpServer?.server_name || "MCP Server Details"}
        width={1000}
        visible={isMcpModalVisible}
        footer={null}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        {selectedMcpServer && (
          <div className="space-y-6">
            {/* Server Overview */}
            <div>
              <Text className="text-lg font-semibold mb-4">Server Overview</Text>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <Text className="font-medium">Server Name:</Text>
                  <Text>{selectedMcpServer.server_name}</Text>
                </div>
                <div>
                  <Text className="font-medium">Server ID:</Text>
                  <div className="flex items-center space-x-2">
                    <Text className="text-xs truncate">{selectedMcpServer.server_id}</Text>
                    <CopyOutlined
                      onClick={() => copyToClipboard(selectedMcpServer.server_id)}
                      className="cursor-pointer text-gray-500 hover:text-blue-500"
                    />
                  </div>
                </div>
                {selectedMcpServer.alias && (
                  <div>
                    <Text className="font-medium">Alias:</Text>
                    <Text>{selectedMcpServer.alias}</Text>
                  </div>
                )}
                <div>
                  <Text className="font-medium">Transport:</Text>
                  <Badge color="blue">{selectedMcpServer.transport}</Badge>
                </div>
                <div>
                  <Text className="font-medium">Auth Type:</Text>
                  <Badge color={selectedMcpServer.auth_type === "none" ? "gray" : "green"}>
                    {selectedMcpServer.auth_type}
                  </Badge>
                </div>
                <div>
                  <Text className="font-medium">Status:</Text>
                  <Badge
                    color={
                      selectedMcpServer.status === "active" || selectedMcpServer.status === "healthy"
                        ? "green"
                        : selectedMcpServer.status === "inactive" || selectedMcpServer.status === "unhealthy"
                          ? "red"
                          : "gray"
                    }
                  >
                    {selectedMcpServer.status || "unknown"}
                  </Badge>
                </div>
              </div>
              {selectedMcpServer.description && (
                <div className="mt-2">
                  <Text className="font-medium">Description:</Text>
                  <Text className="mt-1">{selectedMcpServer.description}</Text>
                </div>
              )}
            </div>

            {/* Connection Details */}
            <div>
              <Text className="text-lg font-semibold mb-4">Connection Details</Text>
              <div className="space-y-2">
                <div>
                  <Text className="font-medium">URL:</Text>
                  <div className="flex items-center space-x-2 mt-1">
                    <Text className="text-sm break-all bg-gray-100 p-2 rounded flex-1">{selectedMcpServer.url}</Text>
                    <CopyOutlined
                      onClick={() => copyToClipboard(selectedMcpServer.url)}
                      className="cursor-pointer text-gray-500 hover:text-blue-500 flex-shrink-0"
                    />
                  </div>
                </div>
                {selectedMcpServer.command && (
                  <div>
                    <Text className="font-medium">Command:</Text>
                    <Text className="text-sm bg-gray-100 p-2 rounded mt-1 font-mono">{selectedMcpServer.command}</Text>
                  </div>
                )}
              </div>
            </div>

            {/* Tools */}
            {selectedMcpServer.allowed_tools && selectedMcpServer.allowed_tools.length > 0 && (
              <div>
                <Text className="text-lg font-semibold mb-4">Allowed Tools</Text>
                <div className="flex flex-wrap gap-2">
                  {selectedMcpServer.allowed_tools.map((tool, idx) => (
                    <Badge key={idx} color="purple">
                      {tool}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Teams */}
            {selectedMcpServer.teams && selectedMcpServer.teams.length > 0 && (
              <div>
                <Text className="text-lg font-semibold mb-4">Teams</Text>
                <div className="flex flex-wrap gap-2">
                  {selectedMcpServer.teams.map((team, idx) => (
                    <Badge key={idx} color="blue">
                      {team}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Access Groups */}
            {selectedMcpServer.mcp_access_groups && selectedMcpServer.mcp_access_groups.length > 0 && (
              <div>
                <Text className="text-lg font-semibold mb-4">Access Groups</Text>
                <div className="flex flex-wrap gap-2">
                  {selectedMcpServer.mcp_access_groups.map((group, idx) => (
                    <Badge key={idx} color="green">
                      {group}
                    </Badge>
                  ))}
                </div>
              </div>
            )}

            {/* Metadata */}
            <div>
              <Text className="text-lg font-semibold mb-4">Metadata</Text>
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <Text className="font-medium">Created By:</Text>
                  <Text>{selectedMcpServer.created_by}</Text>
                </div>
                <div>
                  <Text className="font-medium">Updated By:</Text>
                  <Text>{selectedMcpServer.updated_by}</Text>
                </div>
                <div>
                  <Text className="font-medium">Created At:</Text>
                  <Text className="text-sm">{new Date(selectedMcpServer.created_at).toLocaleString()}</Text>
                </div>
                <div>
                  <Text className="font-medium">Updated At:</Text>
                  <Text className="text-sm">{new Date(selectedMcpServer.updated_at).toLocaleString()}</Text>
                </div>
                {selectedMcpServer.last_health_check && (
                  <div>
                    <Text className="font-medium">Last Health Check:</Text>
                    <Text className="text-sm">{new Date(selectedMcpServer.last_health_check).toLocaleString()}</Text>
                  </div>
                )}
              </div>
              {selectedMcpServer.health_check_error && (
                <div className="mt-2 p-2 bg-red-50 rounded">
                  <Text className="font-medium text-red-700">Health Check Error:</Text>
                  <Text className="text-sm text-red-600 mt-1">{selectedMcpServer.health_check_error}</Text>
                </div>
              )}
            </div>

            {/* Usage Example */}
            <div>
              <Text className="text-lg font-semibold mb-4">Usage Example</Text>
              <SyntaxHighlighter language="python" className="text-sm">
                {`from fastmcp import Client
import asyncio

# Standard MCP configuration
config = {
    "mcpServers": {
        "${selectedMcpServer.server_name}": {
            "url": "http://localhost:4000/${selectedMcpServer.server_name}/mcp",
            "headers": {
                "x-litellm-api-key": "Bearer sk-1234"
            }
        }
    }
}

# Create a client that connects to the server
client = Client(config)

async def main():
    async with client:
        # List available tools
        tools = await client.list_tools()
        print(f"Available tools: {[tool.name for tool in tools]}")

        # Call a tool
        response = await client.call_tool(
            name="tool_name", 
            arguments={"arg": "value"}
        )
        print(f"Response: {response}")

if __name__ == "__main__":
    asyncio.run(main())`}
              </SyntaxHighlighter>
            </div>
          </div>
        )}
      </Modal>

      {/* Make Model Public Form */}
      <MakeModelPublicForm
        visible={isMakePublicModalVisible}
        onClose={() => setIsMakePublicModalVisible(false)}
        accessToken={accessToken || ""}
        modelHubData={modelHubData || []}
        onSuccess={handleMakePublicSuccess}
      />

      {/* Make Agent Public Form */}
      <MakeAgentPublicForm
        visible={isMakeAgentPublicModalVisible}
        onClose={() => setIsMakeAgentPublicModalVisible(false)}
        accessToken={accessToken || ""}
        agentHubData={agentHubData || []}
        onSuccess={handleMakeAgentPublicSuccess}
      />

      {/* Make MCP Public Form */}
      <MakeMCPPublicForm
        visible={isMakeMcpPublicModalVisible}
        onClose={() => setIsMakeMcpPublicModalVisible(false)}
        accessToken={accessToken || ""}
        mcpHubData={mcpHubData || []}
        onSuccess={handleMakeMcpPublicSuccess}
      />
    </div>
  );
};

export default ModelHubTable;
