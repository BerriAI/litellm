import { AgentHubData, getAgentHubTableColumns } from "@/components/AIHub/AgentHubTableColumns";
import MakeAgentPublicForm from "@/components/AIHub/forms/MakeAgentPublicForm";
import MakeMCPPublicForm from "@/components/AIHub/forms/MakeMCPPublicForm";
import MakeModelPublicForm from "@/components/AIHub/forms/MakeModelPublicForm";
import { mcpHubColumns, MCPServerData } from "@/components/mcp_hub_table_columns";
import { modelHubColumns } from "@/components/model_hub_table_columns";
import UsefulLinksManagement from "@/components/AIHub/UsefulLinksManagement";
import { getClaudeCodePluginsList } from "@/components/networking";
import { Plugin } from "@/components/claude_code_plugins/types";
import SkillHubDashboard from "@/components/AIHub/SkillHubDashboard";
import MakeSkillPublicForm from "@/components/claude_code_plugins/MakeSkillPublicForm";
import { ModelDataTable } from "@/components/model_dashboard/table";
import ModelFilters from "@/components/model_filters";
import NotificationsManager from "@/components/molecules/notifications_manager";
import {
  fetchMCPServers,
  getAgentsList,
  getConfigFieldSetting,
  getProxyBaseUrl,
  getUiConfig,
  modelHubCall,
  modelHubPublicModelsCall,
} from "@/components/networking";
import PublicModelHub from "@/components/public_model_hub";
import { isAdminRole } from "@/utils/roles";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { cn } from "@/lib/utils";
import { Copy } from "lucide-react";
import { useRouter } from "next/navigation";
import React, { useCallback, useEffect, useState } from "react";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { checkTokenValidity } from "@/utils/jwtUtils";
import { getCookie } from "@/utils/cookieUtils";

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

const BADGE_COLOR_CLASSES: Record<string, string> = {
  blue: "bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300",
  green:
    "bg-emerald-100 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  purple:
    "bg-purple-100 text-purple-700 dark:bg-purple-950 dark:text-purple-300",
  orange:
    "bg-orange-100 text-orange-700 dark:bg-orange-950 dark:text-orange-300",
  red: "bg-red-100 text-red-700 dark:bg-red-950 dark:text-red-300",
  yellow:
    "bg-amber-100 text-amber-700 dark:bg-amber-950 dark:text-amber-300",
  gray: "bg-muted text-muted-foreground",
};

const colorBadge = (
  color: keyof typeof BADGE_COLOR_CLASSES | string,
  extra?: string,
) =>
  cn(BADGE_COLOR_CLASSES[color] ?? BADGE_COLOR_CLASSES.gray, extra);

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
  // Skill Hub state
  const [skillHubData, setSkillHubData] = useState<Plugin[]>([]);
  const [skillLoading, setSkillLoading] = useState<boolean>(false);
  const [isMakeSkillPublicModalVisible, setIsMakeSkillPublicModalVisible] = useState(false);
  const router = useRouter();
  const { data: uiSettings, isLoading: isUISettingsLoading } = useUISettings();

  // Check authentication requirement for public AI Hub
  useEffect(() => {
    // Only check when UI settings are loaded and this is a public page
    if (isUISettingsLoading || !publicPage) {
      return;
    }

    const requireAuth = uiSettings?.values?.require_auth_for_public_ai_hub;

    // If require_auth_for_public_ai_hub is true, verify token
    if (requireAuth === true) {
      const token = getCookie("token");
      const isTokenValid = checkTokenValidity(token);

      // If token is invalid, redirect to login
      if (!isTokenValid) {
        router.replace(`${getProxyBaseUrl()}/ui/login`);
        return;
      }
    }
    // If require_auth_for_public_ai_hub is false, allow public access (no change)
  }, [isUISettingsLoading, publicPage, uiSettings, router]);

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

  // Fetch Skill Hub data — all skills for admins, enabled-only for public page
  useEffect(() => {
    const fetchSkillData = async () => {
      if (!accessToken) return;
      try {
        setSkillLoading(true);
        const enabledOnly = publicPage === true;
        const response = await getClaudeCodePluginsList(accessToken, enabledOnly);
        setSkillHubData(response.plugins);
      } catch (error) {
        console.error("Error fetching skill hub data", error);
      } finally {
        setSkillLoading(false);
      }
    };

    fetchSkillData();
  }, [accessToken, publicPage]);

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
    setIsMakePublicModalVisible(true);
  };

  const handleMakeAgentPublicPage = () => {
    if (!accessToken) {
      return;
    }
    setIsMakeAgentPublicModalVisible(true);
  };

  const handleMakeMcpPublicPage = () => {
    if (!accessToken) {
      return;
    }
    setIsMakeMcpPublicModalVisible(true);
  };

  const closeDetailDialogs = () => {
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
    return key
      .replace(/^supports_/, "")
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  const getModelCapabilities = (model: ModelGroupInfo) => {
    return Object.entries(model)
      .filter(([key, value]) => key.startsWith("supports_") && value === true)
      .map(([key]) => key);
  };

  const formatCost = (cost: number) => {
    return `$${(cost * 1_000_000).toFixed(2)}`;
  };

  const handleMakePublicSuccess = () => {
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
              <h2 className="text-2xl font-semibold m-0">AI Hub</h2>
              {isAdminRole(userRole || "") ? (
                <p className="text-sm text-muted-foreground">
                  Make models, agents, and MCP servers public for developers to know what&apos;s available.
                </p>
              ) : (
                <p className="text-sm text-muted-foreground">A list of all public model names personally available to you.</p>
              )}
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm">Model Hub URL:</span>
              <div className="flex items-center bg-muted px-2 py-1 rounded">
                <span className="mr-2 text-sm">{`${getProxyBaseUrl()}/ui/model_hub_table`}</span>
                <button
                  onClick={() => copyToClipboard(`${getProxyBaseUrl()}/ui/model_hub_table`)}
                  className="p-1 hover:bg-muted-foreground/10 rounded transition-colors"
                  title="Copy URL"
                >
                  <Copy size={16} className="text-muted-foreground" />
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

          {/* Tab System for Model Hub, Agent Hub, MCP Hub, and Plugin Marketplace */}
          <Tabs defaultValue="models" className="w-full">
            <TabsList className="mb-4">
              <TabsTrigger value="models">Model Hub</TabsTrigger>
              <TabsTrigger value="agents">Agent Hub</TabsTrigger>
              <TabsTrigger value="mcp">MCP Hub</TabsTrigger>
              <TabsTrigger value="skills">Skill Hub</TabsTrigger>
            </TabsList>

            {/* Model Hub Tab */}
            <TabsContent value="models">
              <Card className="p-6">
                {publicPage == false && isAdminRole(userRole || "") && (
                  <div className="flex justify-end mb-4">
                    <Button onClick={() => handleMakePublicPage()}>Select Models to Make Public</Button>
                  </div>
                )}

                <ModelFilters modelHubData={modelHubData || []} onFilteredDataChange={handleFilteredDataChange} />

                <ModelDataTable
                  columns={modelHubColumns(showModal, copyToClipboard, publicPage)}
                  data={filteredData}
                  isLoading={loading}
                  defaultSorting={[{ id: "model_group", desc: false }]}
                />
              </Card>

              <div className="mt-4 text-center space-y-2">
                <span className="text-sm text-muted-foreground">
                  Showing {filteredData.length} of {modelHubData?.length || 0} models
                </span>
              </div>
            </TabsContent>

            {/* Agent Hub Tab */}
            <TabsContent value="agents">
              <Card className="p-6">
                {publicPage == false && isAdminRole(userRole || "") && (
                  <div className="flex justify-end mb-4">
                    <Button onClick={() => handleMakeAgentPublicPage()}>Select Agents to Make Public</Button>
                  </div>
                )}

                <ModelDataTable
                  columns={getAgentHubTableColumns(showAgentModal, copyToClipboard, publicPage)}
                  data={agentHubData || []}
                  isLoading={agentLoading}
                  defaultSorting={[{ id: "name", desc: false }]}
                />
              </Card>

              <div className="mt-4 text-center space-y-2">
                <span className="text-sm text-muted-foreground">
                  Showing {agentHubData?.length || 0} agent{agentHubData?.length !== 1 ? "s" : ""}
                </span>
              </div>
            </TabsContent>

            {/* MCP Hub Tab */}
            <TabsContent value="mcp">
              <Card className="p-6">
                {publicPage == false && isAdminRole(userRole || "") && (
                  <div className="flex justify-end mb-4">
                    <Button onClick={() => handleMakeMcpPublicPage()}>Select MCP Servers to Make Public</Button>
                  </div>
                )}

                <ModelDataTable
                  columns={mcpHubColumns(showMcpModal, copyToClipboard, publicPage)}
                  data={mcpHubData || []}
                  isLoading={mcpLoading}
                  defaultSorting={[{ id: "server_name", desc: false }]}
                />
              </Card>

              <div className="mt-4 text-center space-y-2">
                <span className="text-sm text-muted-foreground">
                  Showing {mcpHubData?.length || 0} MCP server{mcpHubData?.length !== 1 ? "s" : ""}
                </span>
              </div>
            </TabsContent>

            {/* Skill Hub Tab */}
            <TabsContent value="skills">
              {publicPage == false && isAdminRole(userRole || "") && (
                <div className="flex justify-end mb-4">
                  <Button onClick={() => setIsMakeSkillPublicModalVisible(true)}>
                    Select Skills to Make Public
                  </Button>
                </div>
              )}
              <SkillHubDashboard
                skills={skillHubData}
                isLoading={skillLoading}
                isAdmin={isAdminRole(userRole || "")}
                accessToken={accessToken}
                publicPage={publicPage}
                onPublishSuccess={async () => {
                  const response = await getClaudeCodePluginsList(accessToken || "", publicPage);
                  setSkillHubData(response.plugins);
                }}
              />
            </TabsContent>
          </Tabs>
        </div>
      ) : (
        <Card className="mx-auto max-w-xl mt-10 p-6">
          <p className="text-xl text-center mb-2 text-foreground">Public Model Hub not enabled.</p>
          <p className="text-base text-center text-muted-foreground">
            Ask your proxy admin to enable this on their Admin UI.
          </p>
        </Card>
      )}

      {/* Public Page Modal */}
      <Dialog
        open={isPublicPageModalVisible}
        onOpenChange={(o) => (!o ? closeDetailDialogs() : undefined)}
      >
        <DialogContent className="max-w-[600px]">
          <DialogHeader>
            <DialogTitle>Public Model Hub</DialogTitle>
          </DialogHeader>
          <div className="pt-5 pb-5">
            <div className="flex justify-between mb-4">
              <span className="text-base mr-2">Shareable Link:</span>
              <span className="max-w-sm ml-2 bg-muted pr-2 pl-2 pt-1 pb-1 text-center rounded text-sm">
                {`${getProxyBaseUrl()}/ui/model_hub_table`}
              </span>
            </div>
            <div className="flex justify-end">
              <Button onClick={goToPublicModelPage}>See Page</Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Model Details Modal */}
      <Dialog
        open={isModalVisible}
        onOpenChange={(o) => (!o ? closeDetailDialogs() : undefined)}
      >
        <DialogContent className="max-w-[1000px] max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{selectedModel?.model_group || "Model Details"}</DialogTitle>
          </DialogHeader>
          {selectedModel && (
            <div className="space-y-6">
              {/* Model Overview */}
              <div>
                <p className="text-lg font-semibold mb-4">Model Overview</p>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="font-medium">Model Group:</p>
                    <p>{selectedModel.model_group}</p>
                  </div>
                  <div>
                    <p className="font-medium">Mode:</p>
                    <p>{selectedModel.mode || "Not specified"}</p>
                  </div>
                  <div>
                    <p className="font-medium">Providers:</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {selectedModel.providers.map((provider) => (
                        <Badge key={provider} className={colorBadge("blue")}>
                          {provider}
                        </Badge>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              {/* Token and Cost Information */}
              <div>
                <p className="text-lg font-semibold mb-4">Token & Cost Information</p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="font-medium">Max Input Tokens:</p>
                    <p>{selectedModel.max_input_tokens?.toLocaleString() || "Not specified"}</p>
                  </div>
                  <div>
                    <p className="font-medium">Max Output Tokens:</p>
                    <p>{selectedModel.max_output_tokens?.toLocaleString() || "Not specified"}</p>
                  </div>
                  <div>
                    <p className="font-medium">Input Cost per 1M Tokens:</p>
                    <p>
                      {selectedModel.input_cost_per_token
                        ? formatCost(selectedModel.input_cost_per_token)
                        : "Not specified"}
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">Output Cost per 1M Tokens:</p>
                    <p>
                      {selectedModel.output_cost_per_token
                        ? formatCost(selectedModel.output_cost_per_token)
                        : "Not specified"}
                    </p>
                  </div>
                </div>
              </div>

              {/* Capabilities */}
              <div>
                <p className="text-lg font-semibold mb-4">Capabilities</p>
                <div className="flex flex-wrap gap-2">
                  {(() => {
                    const capabilities = getModelCapabilities(selectedModel);
                    const colors = ["green", "blue", "purple", "orange", "red", "yellow"];

                    if (capabilities.length === 0) {
                      return <p className="text-muted-foreground">No special capabilities listed</p>;
                    }

                    return capabilities.map((capability, index) => (
                      <Badge key={capability} className={colorBadge(colors[index % colors.length])}>
                        {formatCapabilityName(capability)}
                      </Badge>
                    ));
                  })()}
                </div>
              </div>

              {/* Rate Limits */}
              {(selectedModel.tpm || selectedModel.rpm) && (
                <div>
                  <p className="text-lg font-semibold mb-4">Rate Limits</p>
                  <div className="grid grid-cols-2 gap-4">
                    {selectedModel.tpm && (
                      <div>
                        <p className="font-medium">Tokens per Minute:</p>
                        <p>{selectedModel.tpm.toLocaleString()}</p>
                      </div>
                    )}
                    {selectedModel.rpm && (
                      <div>
                        <p className="font-medium">Requests per Minute:</p>
                        <p>{selectedModel.rpm.toLocaleString()}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Supported OpenAI Parameters */}
              {selectedModel.supported_openai_params && (
                <div>
                  <p className="text-lg font-semibold mb-4">Supported OpenAI Parameters</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedModel.supported_openai_params.map((param) => (
                      <Badge key={param} className={colorBadge("green")}>
                        {param}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Usage Example */}
              <div>
                <p className="text-lg font-semibold mb-4">Usage Example</p>
                <SyntaxHighlighter language="python" className="text-sm">
                  {`import openai

client = openai.OpenAI(
    api_key="your_api_key",
    base_url="${getProxyBaseUrl()}"  # Your LiteLLM Proxy URL
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
        </DialogContent>
      </Dialog>

      {/* Agent Details Modal */}
      <Dialog
        open={isAgentModalVisible}
        onOpenChange={(o) => (!o ? closeDetailDialogs() : undefined)}
      >
        <DialogContent className="max-w-[1000px] max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{selectedAgent?.name || "Agent Details"}</DialogTitle>
          </DialogHeader>
          {selectedAgent && (
            <div className="space-y-6">
              {/* Agent Overview */}
              <div>
                <p className="text-lg font-semibold mb-4">Agent Overview</p>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="font-medium">Name:</p>
                    <p>{selectedAgent.name}</p>
                  </div>
                  <div>
                    <p className="font-medium">Version:</p>
                    <Badge className={colorBadge("blue")}>v{selectedAgent.version}</Badge>
                  </div>
                  <div>
                    <p className="font-medium">Protocol Version:</p>
                    <p>{selectedAgent.protocolVersion}</p>
                  </div>
                  <div>
                    <p className="font-medium">URL:</p>
                    <div className="flex items-center space-x-2">
                      <span className="truncate">{selectedAgent.url}</span>
                      <Copy
                        size={14}
                        onClick={() => copyToClipboard(selectedAgent.url)}
                        className="cursor-pointer text-muted-foreground hover:text-primary"
                      />
                    </div>
                  </div>
                </div>
                <div>
                  <p className="font-medium">Description:</p>
                  <p className="mt-1">{selectedAgent.description}</p>
                </div>
              </div>

              {/* Capabilities */}
              {selectedAgent.capabilities && Object.keys(selectedAgent.capabilities).length > 0 && (
                <div>
                  <p className="text-lg font-semibold mb-4">Capabilities</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(selectedAgent.capabilities)
                      .filter(([_, value]) => value === true)
                      .map(([key]) => (
                        <Badge key={key} className={colorBadge("green")}>
                          {key}
                        </Badge>
                      ))}
                  </div>
                </div>
              )}

              {/* Input/Output Modes */}
              <div>
                <p className="text-lg font-semibold mb-4">Input/Output Modes</p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="font-medium">Input Modes:</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {selectedAgent.defaultInputModes?.map((mode) => (
                        <Badge key={mode} className={colorBadge("blue")}>
                          {mode}
                        </Badge>
                      )) || <p>Not specified</p>}
                    </div>
                  </div>
                  <div>
                    <p className="font-medium">Output Modes:</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {selectedAgent.defaultOutputModes?.map((mode) => (
                        <Badge key={mode} className={colorBadge("purple")}>
                          {mode}
                        </Badge>
                      )) || <p>Not specified</p>}
                    </div>
                  </div>
                </div>
              </div>

              {/* Skills */}
              {selectedAgent.skills && selectedAgent.skills.length > 0 && (
                <div>
                  <p className="text-lg font-semibold mb-4">Skills</p>
                  <div className="space-y-4">
                    {selectedAgent.skills.map((skill) => (
                      <div key={skill.id} className="border border-border rounded p-4">
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <p className="font-medium text-base">{skill.name}</p>
                            <p className="text-xs text-muted-foreground">ID: {skill.id}</p>
                          </div>
                          {skill.tags && skill.tags.length > 0 && (
                            <div className="flex flex-wrap gap-1">
                              {skill.tags.map((tag) => (
                                <Badge key={tag} className={colorBadge("purple", "text-[10px]")}>
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                        <p className="text-sm mb-2">{skill.description}</p>
                        {skill.examples && skill.examples.length > 0 && (
                          <div>
                            <p className="text-xs font-medium text-foreground">Examples:</p>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {skill.examples.map((example, idx) => (
                                <Badge key={idx} className={colorBadge("gray", "text-[10px]")}>
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
                  <p className="text-lg font-semibold mb-4">Additional Features</p>
                  <Badge className={colorBadge("green")}>Supports Authenticated Extended Card</Badge>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* MCP Server Details Modal */}
      <Dialog
        open={isMcpModalVisible}
        onOpenChange={(o) => (!o ? closeDetailDialogs() : undefined)}
      >
        <DialogContent className="max-w-[1000px] max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle>{selectedMcpServer?.server_name || "MCP Server Details"}</DialogTitle>
          </DialogHeader>
          {selectedMcpServer && (
            <div className="space-y-6">
              {/* Server Overview */}
              <div>
                <p className="text-lg font-semibold mb-4">Server Overview</p>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="font-medium">Server Name:</p>
                    <p>{selectedMcpServer.server_name}</p>
                  </div>
                  <div>
                    <p className="font-medium">Server ID:</p>
                    <div className="flex items-center space-x-2">
                      <span className="text-xs truncate">{selectedMcpServer.server_id}</span>
                      <Copy
                        size={14}
                        onClick={() => copyToClipboard(selectedMcpServer.server_id)}
                        className="cursor-pointer text-muted-foreground hover:text-primary"
                      />
                    </div>
                  </div>
                  {selectedMcpServer.alias && (
                    <div>
                      <p className="font-medium">Alias:</p>
                      <p>{selectedMcpServer.alias}</p>
                    </div>
                  )}
                  <div>
                    <p className="font-medium">Transport:</p>
                    <Badge className={colorBadge("blue")}>{selectedMcpServer.transport}</Badge>
                  </div>
                  <div>
                    <p className="font-medium">Auth Type:</p>
                    <Badge
                      className={colorBadge(
                        selectedMcpServer.auth_type === "none" ? "gray" : "green",
                      )}
                    >
                      {selectedMcpServer.auth_type}
                    </Badge>
                  </div>
                  <div>
                    <p className="font-medium">Status:</p>
                    <Badge
                      className={colorBadge(
                        selectedMcpServer.status === "active" || selectedMcpServer.status === "healthy"
                          ? "green"
                          : selectedMcpServer.status === "inactive" || selectedMcpServer.status === "unhealthy"
                            ? "red"
                            : "gray",
                      )}
                    >
                      {selectedMcpServer.status || "unknown"}
                    </Badge>
                  </div>
                </div>
                {selectedMcpServer.description && (
                  <div className="mt-2">
                    <p className="font-medium">Description:</p>
                    <p className="mt-1">{selectedMcpServer.description}</p>
                  </div>
                )}
              </div>

              {/* Connection Details */}
              <div>
                <p className="text-lg font-semibold mb-4">Connection Details</p>
                <div className="space-y-2">
                  <div>
                    <p className="font-medium">URL:</p>
                    <div className="flex items-center space-x-2 mt-1">
                      <span className="text-sm break-all bg-muted p-2 rounded flex-1">{selectedMcpServer.url}</span>
                      <Copy
                        size={14}
                        onClick={() => copyToClipboard(selectedMcpServer.url)}
                        className="cursor-pointer text-muted-foreground hover:text-primary flex-shrink-0"
                      />
                    </div>
                  </div>
                  {selectedMcpServer.command && (
                    <div>
                      <p className="font-medium">Command:</p>
                      <p className="text-sm bg-muted p-2 rounded mt-1 font-mono">{selectedMcpServer.command}</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Tools */}
              {selectedMcpServer.allowed_tools && selectedMcpServer.allowed_tools.length > 0 && (
                <div>
                  <p className="text-lg font-semibold mb-4">Allowed Tools</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedMcpServer.allowed_tools.map((tool, idx) => (
                      <Badge key={idx} className={colorBadge("purple")}>
                        {tool}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Teams */}
              {selectedMcpServer.teams && selectedMcpServer.teams.length > 0 && (
                <div>
                  <p className="text-lg font-semibold mb-4">Teams</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedMcpServer.teams.map((team, idx) => (
                      <Badge key={idx} className={colorBadge("blue")}>
                        {team}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Access Groups */}
              {selectedMcpServer.mcp_access_groups && selectedMcpServer.mcp_access_groups.length > 0 && (
                <div>
                  <p className="text-lg font-semibold mb-4">Access Groups</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedMcpServer.mcp_access_groups.map((group, idx) => (
                      <Badge key={idx} className={colorBadge("green")}>
                        {group}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Metadata */}
              <div>
                <p className="text-lg font-semibold mb-4">Metadata</p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="font-medium">Created By:</p>
                    <p>{selectedMcpServer.created_by}</p>
                  </div>
                  <div>
                    <p className="font-medium">Updated By:</p>
                    <p>{selectedMcpServer.updated_by}</p>
                  </div>
                  <div>
                    <p className="font-medium">Created At:</p>
                    <p className="text-sm">{new Date(selectedMcpServer.created_at).toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="font-medium">Updated At:</p>
                    <p className="text-sm">{new Date(selectedMcpServer.updated_at).toLocaleString()}</p>
                  </div>
                  {selectedMcpServer.last_health_check && (
                    <div>
                      <p className="font-medium">Last Health Check:</p>
                      <p className="text-sm">{new Date(selectedMcpServer.last_health_check).toLocaleString()}</p>
                    </div>
                  )}
                </div>
                {selectedMcpServer.health_check_error && (
                  <div className="mt-2 p-2 bg-red-50 dark:bg-red-950/30 rounded">
                    <p className="font-medium text-red-700 dark:text-red-300">Health Check Error:</p>
                    <p className="text-sm text-red-600 dark:text-red-400 mt-1">{selectedMcpServer.health_check_error}</p>
                  </div>
                )}
              </div>

              {/* Usage Example */}
              <div>
                <p className="text-lg font-semibold mb-4">Usage Example</p>
                <SyntaxHighlighter language="python" className="text-sm">
                  {`from fastmcp import Client
import asyncio

# Standard MCP configuration
config = {
    "mcpServers": {
        "${selectedMcpServer.server_name}": {
            "url": "${getProxyBaseUrl()}/${selectedMcpServer.server_name}/mcp",
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
        </DialogContent>
      </Dialog>

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

      {/* Make Skill Public Form */}
      <MakeSkillPublicForm
        visible={isMakeSkillPublicModalVisible}
        onClose={() => setIsMakeSkillPublicModalVisible(false)}
        accessToken={accessToken || ""}
        skillsList={skillHubData}
        onSuccess={async () => {
          const response = await getClaudeCodePluginsList(accessToken || "", publicPage === true);
          setSkillHubData(response.plugins);
        }}
      />
    </div>
  );
};

export default ModelHubTable;
