import { useTerminology } from "@/contexts/TerminologyContext";
import i18n from "@/i18n/i18n";
import { AgentHubData, getAgentHubTableColumns } from "@/components/AIHub/AgentHubTableColumns";
import MakeAgentPublicForm from "@/components/AIHub/forms/MakeAgentPublicForm";
import MakeMCPPublicForm from "@/components/AIHub/forms/MakeMCPPublicForm";
import MakeModelPublicForm from "@/components/AIHub/forms/MakeModelPublicForm";
import { getMCPHubTableColumns, MCPServerData } from "@/components/AIHub/MCPHubTableColumns";
import { getModelHubTableColumns, ModelHubData } from "@/components/AIHub/ModelHubTableColumns";
import { modelUsageExample } from "@/components/AIHub/modelUsageExample";
import UsefulLinksManagement from "@/components/AIHub/UsefulLinksManagement";
import { getClaudeCodePluginsList } from "@/components/networking";
import { Plugin } from "@/components/claude_code_plugins/types";
import SkillHubDashboard from "@/components/AIHub/SkillHubDashboard";
import MakeSkillPublicForm from "@/components/claude_code_plugins/MakeSkillPublicForm";
import { DataTable } from "@/components/shared/DataTable";
import ModelFilters from "@/components/model_filters";
import {
  fetchMCPServers,
  getGatewayBaseUrl,
  getAgentsList,
  getConfigFieldSetting,
  getGlobalLitellmHeaderName,
  getProxyBaseUrl,
  getUiPageUrl,
  getUiConfig,
  modelHubCall,
  modelHubPublicModelsCall,
} from "@/components/networking";
import PublicModelHub from "@/components/public_model_hub";
import { copyToClipboard } from "@/utils/dataUtils";
import { isAdminRole, isProxyAdminRole } from "@/utils/roles";
import { filterBySearchTerm } from "@/utils/searchUtils";
import { SortingState } from "@tanstack/react-table";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { InputGroup, InputGroupAddon, InputGroupButton, InputGroupInput } from "@/components/ui/input-group";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Copy, Inbox, Search as SearchIcon, X } from "lucide-react";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { prism } from "react-syntax-highlighter/dist/esm/styles/prism";

import { useSyntaxTheme } from "@/hooks/useSyntaxTheme";
import { useUISettings } from "@/app/(dashboard)/hooks/uiSettings/useUISettings";
import { checkTokenValidity } from "@/utils/jwtUtils";
import { getCookie } from "@/utils/cookieUtils";
import { getLoginUrl } from "@/utils/returnUrlUtils";
import { uiHref } from "@/utils/uiHref";
import {
  DECISIONS_DOCS_URL,
  DECISIONS_MODE,
  SYSTEM_ONE_PLAYGROUND_ROUTE,
  displayMode,
  isDecisionMode,
} from "@/lib/decisionModels";
import DecisionModelGarden from "@/components/AIHub/DecisionModelGarden";
import { DeploymentPricingBreakdown } from "@/components/molecules/models/DeploymentPricingBreakdown";
import { ModelCustomMetadata } from "./ModelCustomMetadata";

interface ModelHubTableProps {
  accessToken: string | null;
  publicPage: boolean;
  premiumUser: boolean;
  userRole: string | null;
  canOpenPlayground: boolean;
  canAddModels: boolean;
  variant?: "dashboard" | "chat";
}

function isMCPHubVisibilityDisabled(isLoading: boolean, servers: readonly MCPServerData[] | null): boolean {
  return isLoading || servers === null;
}

function modelTableEmptyState(
  models: readonly ModelHubData[],
  showsDecisionGarden: boolean,
  selectedMode: string,
): { title: string; body: string } {
  if (showsDecisionGarden && selectedMode === DECISIONS_MODE && !models.some((model) => isDecisionMode(model.mode))) {
    return {
      title: i18n.t("aiHub.modelHubTable.models.empty.decisionsTitle"),
      body: i18n.t("aiHub.modelHubTable.models.empty.decisionsBody"),
    };
  }
  return models.length > 0
    ? {
        title: i18n.t("aiHub.modelHubTable.models.empty.noMatchTitle"),
        body: i18n.t("aiHub.modelHubTable.models.empty.noMatchBody"),
      }
    : {
        title: i18n.t("aiHub.modelHubTable.models.empty.noneTitle"),
        body: i18n.t("aiHub.modelHubTable.models.empty.noneBody"),
      };
}

function HubEmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex flex-col items-center gap-1 py-6">
      <div className="mb-1 flex size-10 items-center justify-center rounded-lg bg-muted">
        <Inbox className="size-5 text-muted-foreground" />
      </div>
      <div className="text-sm font-medium text-foreground">{title}</div>
      <div className="text-sm text-muted-foreground">{body}</div>
    </div>
  );
}

const ModelHubTable: React.FC<ModelHubTableProps> = ({
  accessToken,
  publicPage,
  premiumUser,
  userRole,
  canOpenPlayground,
  canAddModels,
  variant = "dashboard",
}) => {
  const term = useTerminology();
  const { t } = useTranslation();
  const syntaxTheme = useSyntaxTheme(prism);
  const isChat = variant === "chat";
  // Admin Viewer follows the read-parity rule: see the AI Hub catalog, but
  // cannot toggle public visibility (write).
  const canModify = !isChat && isProxyAdminRole(userRole || "");

  const [publicPageAllowed, setPublicPageAllowed] = useState<boolean>(false);
  const [modelHubData, setModelHubData] = useState<ModelHubData[] | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedModel, setSelectedModel] = useState<null | ModelHubData>(null);
  const [filteredData, setFilteredData] = useState<ModelHubData[]>([]);
  const [selectedMode, setSelectedMode] = useState<string>("");
  const [selectedProvider, setSelectedProvider] = useState("");
  const isDashboardCatalog = !isChat && !publicPage;
  const showDecisionGarden = isDashboardCatalog && (selectedMode === "" || selectedMode === DECISIONS_MODE);
  const [isMakePublicModalVisible, setIsMakePublicModalVisible] = useState(false);
  // Agent Hub state
  const [agentHubData, setAgentHubData] = useState<AgentHubData[] | null>(null);
  const [isMakeAgentPublicModalVisible, setIsMakeAgentPublicModalVisible] = useState(false);
  const [agentLoading, setAgentLoading] = useState<boolean>(true);
  const [selectedAgent, setSelectedAgent] = useState<null | AgentHubData>(null);
  const [isAgentModalVisible, setIsAgentModalVisible] = useState(false);
  const [agentSearchTerm, setAgentSearchTerm] = useState("");
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
        window.location.replace(getUiPageUrl(getLoginUrl(getProxyBaseUrl())));
        return;
      }
    }
    // If require_auth_for_public_ai_hub is false, allow public access (no change)
  }, [isUISettingsLoading, publicPage, uiSettings]);

  useEffect(() => {
    const fetchData = async (accessToken: string) => {
      try {
        setLoading(true);
        const _modelHubData = await modelHubCall(accessToken);
        setModelHubData(_modelHubData.data);

        if (!isChat) {
          getConfigFieldSetting(accessToken, "enable_public_model_hub")
            .then((data) => {
              if (data.field_value == true) {
                setPublicPageAllowed(true);
              }
            })
            .catch((error) => {
              // do nothing
            });
        }
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
        setModelHubData(_modelHubData);
        setPublicPageAllowed(true);
      } catch (error) {
        console.error("There was an error fetching the public model data", error);
      } finally {
        setLoading(false);
      }
    };

    const fetchModelData = async () => {
      if (accessToken) {
        await fetchData(accessToken);
      } else if (publicPage) {
        await fetchPublicData();
      } else {
        setLoading(false);
      }
    };
    fetchModelData();
  }, [accessToken, publicPage, isChat]);

  // Fetch Agent Hub data
  useEffect(() => {
    const fetchAgentData = async () => {
      if (!accessToken) {
        setAgentLoading(false);
        return;
      }

      try {
        setAgentLoading(true);
        const response = await getAgentsList(accessToken);
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

    if (!publicPage && !isChat) {
      fetchAgentData();
    }
  }, [publicPage, accessToken, isChat]);

  // Fetch MCP Hub data
  useEffect(() => {
    const fetchMcpData = async () => {
      if (!accessToken) {
        setMcpLoading(false);
        return;
      }

      try {
        setMcpLoading(true);
        const response = await fetchMCPServers(accessToken);
        setMcpHubData(response);
      } catch (error) {
        console.error("There was an error fetching the MCP server data", error);
      } finally {
        setMcpLoading(false);
      }
    };

    if (!publicPage && !isChat) {
      fetchMcpData();
    }
  }, [publicPage, accessToken, isChat]);

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

    if (!isChat) fetchSkillData();
  }, [accessToken, publicPage, isChat]);

  const showModal = useCallback((model: ModelHubData) => {
    setSelectedModel(model);
    setIsModalVisible(true);
  }, []);

  const showAgentModal = useCallback((agent: AgentHubData) => {
    setSelectedAgent(agent);
    setIsAgentModalVisible(true);
  }, []);

  const showMcpModal = useCallback((server: MCPServerData) => {
    setSelectedMcpServer(server);
    setIsMcpModalVisible(true);
  }, []);

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
    setSelectedModel(null);
    setIsAgentModalVisible(false);
    setSelectedAgent(null);
    setIsMcpModalVisible(false);
    setSelectedMcpServer(null);
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    setSelectedModel(null);
    setIsAgentModalVisible(false);
    setSelectedAgent(null);
    setIsMcpModalVisible(false);
    setSelectedMcpServer(null);
  };

  const formatCapabilityName = (key: string) => {
    // Remove 'supports_' prefix and convert snake_case to Title Case
    return key
      .replace(/^supports_/, "")
      .split("_")
      .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
      .join(" ");
  };

  const getModelCapabilities = (model: ModelHubData) => {
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
          setMcpLoading(true);
          const response = await fetchMCPServers(accessToken);
          setMcpHubData(response);
        } catch (error) {
          setMcpHubData(null);
          console.error("Error refreshing MCP server data:", error);
        } finally {
          setMcpLoading(false);
        }
      };
      fetchMcpData();
    }
  };

  const handleFilteredDataChange = useCallback((newFilteredData: ModelHubData[]) => {
    setFilteredData(newFilteredData);
  }, []);

  const [modelSorting, setModelSorting] = useState<SortingState>([{ id: "model_group", desc: false }]);
  const [agentSorting, setAgentSorting] = useState<SortingState>([{ id: "name", desc: false }]);
  const [mcpSorting, setMcpSorting] = useState<SortingState>([{ id: "server_name", desc: false }]);

  const modelColumns = useMemo(
    () => getModelHubTableColumns({ onModelClick: showModal, showProviderLogos: isChat }),
    [showModal, isChat],
  );
  const agentColumns = useMemo(() => getAgentHubTableColumns({ onAgentClick: showAgentModal }), [showAgentModal]);
  const filteredAgentData = useMemo(
    () => filterBySearchTerm(agentHubData ?? [], agentSearchTerm, (agent) => [agent.name, agent.description]),
    [agentHubData, agentSearchTerm],
  );
  const mcpColumns = useMemo(() => getMCPHubTableColumns({ onServerClick: showMcpModal }), [showMcpModal]);

  // If this is a public page, use the dedicated PublicModelHub component
  if (publicPage && publicPageAllowed) {
    return <PublicModelHub accessToken={accessToken} />;
  }

  return (
    <div data-slot="model-hub" className={isChat ? "chat-model-hub-catalog" : "mx-4"}>
      {publicPage == false ? (
        <div className={isChat ? "chat-model-hub-content" : "w-full m-2 mt-2 p-8"}>
          {/* Header with Title, Description and URL */}
          {!isChat && (
            <div className="flex justify-between items-center mb-6">
              <div className="flex flex-col items-start">
                <h2 className="text-center text-xl font-semibold">{term(t("aiHub.modelHubTable.title"))}</h2>
                {isAdminRole(userRole || "") ? (
                  <p className="text-sm text-muted-foreground">{t("aiHub.modelHubTable.adminDescription")}</p>
                ) : (
                  <p className="text-sm text-muted-foreground">{t("aiHub.modelHubTable.userDescription")}</p>
                )}
              </div>
              <div className="flex items-center space-x-4">
                <p>{t("aiHub.modelHubTable.hubUrlLabel")}</p>
                <div className="flex items-center bg-border px-2 py-1 rounded-sm">
                  <p className="mr-2">{getUiPageUrl(`${getProxyBaseUrl()}/ui/model_hub_table`)}</p>
                  <button
                    onClick={() => void copyToClipboard(getUiPageUrl(`${getProxyBaseUrl()}/ui/model_hub_table`))}
                    className="p-1 hover:bg-accent rounded-sm transition-colors"
                    title={t("aiHub.modelHubTable.copyUrl")}
                  >
                    <Copy size={16} className="text-muted-foreground" />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Useful Links Management Section for Admins */}
          {canModify && (
            <div className="mt-8 mb-2">
              <UsefulLinksManagement accessToken={accessToken} userRole={userRole} />
            </div>
          )}

          {/* Tab System for Model Hub, Agent Hub, MCP Hub, and Plugin Marketplace */}
          <Tabs defaultValue="models">
            {!isChat && (
              <TabsList variant="line" className="mb-4 h-auto w-full justify-start rounded-none border-b p-0">
                <TabsTrigger value="models" className="flex-none rounded-none px-4 py-2">
                  {t("aiHub.modelHubTable.tabs.models")}
                </TabsTrigger>
                <TabsTrigger value="agents" className="flex-none rounded-none px-4 py-2">
                  {t("aiHub.modelHubTable.tabs.agents")}
                </TabsTrigger>
                <TabsTrigger value="mcp" className="flex-none rounded-none px-4 py-2">
                  {t("aiHub.modelHubTable.tabs.mcp")}
                </TabsTrigger>
                <TabsTrigger value="skills" className="flex-none rounded-none px-4 py-2">
                  {t("aiHub.modelHubTable.tabs.skills")}
                </TabsTrigger>
              </TabsList>
            )}

            <div>
              {/* Model Hub Tab */}
              <TabsContent value="models" keepMounted>
                {/* Model Filters and Table */}
                <Card className={isChat ? "chat-model-hub-card" : "px-6"}>
                  {isChat && (
                    <div className="chat-model-section-heading">
                      <h2>{t("aiHub.modelHubTable.models.chatTitle")}</h2>
                      <span
                        className="chat-model-section-count"
                        aria-label={t("aiHub.modelHubTable.models.chatCount", { shown: filteredData.length })}
                      >
                        {filteredData.length}
                      </span>
                    </div>
                  )}
                  {/* Header with Make Public Button */}
                  {publicPage == false && canModify && (
                    <div className="flex justify-end mb-4">
                      <Button onClick={() => handleMakePublicPage()}>
                        {t("aiHub.modelHubTable.models.makePublic")}
                      </Button>
                    </div>
                  )}

                  {/* Filters */}
                  <ModelFilters
                    variant={variant}
                    modelHubData={modelHubData || []}
                    className={isChat ? "chat-model-hub-filters" : undefined}
                    showFiltersCard={!isChat}
                    onFilteredDataChange={handleFilteredDataChange}
                    extraModes={isDashboardCatalog ? [DECISIONS_MODE] : []}
                    onModeChange={setSelectedMode}
                    selectedProvider={isChat ? selectedProvider : undefined}
                    onProviderChange={isChat ? setSelectedProvider : undefined}
                  />

                  {/* Model Table */}
                  <DataTable
                    data={filteredData}
                    paginationMode="client"
                    columns={modelColumns}
                    getRowId={(model, index) => model.model_group || String(index)}
                    sortingMode="client"
                    sorting={modelSorting}
                    onSortingChange={setModelSorting}
                    isLoading={loading}
                    loadingMessage={t("aiHub.modelHubTable.models.loading")}
                    noDataMessage={
                      <HubEmptyState {...modelTableEmptyState(modelHubData ?? [], showDecisionGarden, selectedMode)} />
                    }
                    size="compact"
                  />
                </Card>

                {!isChat && (
                  <div className="mt-4 text-center space-y-2">
                    <p className="text-sm text-muted-foreground">
                      {t("aiHub.modelHubTable.models.showing", {
                        shown: filteredData.length,
                        total: modelHubData?.length || 0,
                      })}
                    </p>
                  </div>
                )}

                {showDecisionGarden && <DecisionModelGarden accessToken={accessToken} canAddModels={canAddModels} />}
              </TabsContent>

              {/* Agent Hub Tab */}
              <TabsContent value="agents" keepMounted>
                <Card className="px-6">
                  {/* Header with Make Public Button */}
                  {publicPage == false && canModify && (
                    <div className="flex justify-end mb-4">
                      <Button onClick={() => handleMakeAgentPublicPage()}>
                        {t("aiHub.modelHubTable.agents.makePublic")}
                      </Button>
                    </div>
                  )}

                  <div className="mb-4">
                    <p className="text-sm font-medium mb-2">{t("aiHub.modelHubTable.agents.searchLabel")}</p>
                    <InputGroup className="max-w-sm">
                      <InputGroupAddon>
                        <SearchIcon className="size-4 text-muted-foreground" />
                      </InputGroupAddon>
                      <InputGroupInput
                        placeholder={t("aiHub.modelHubTable.agents.searchPlaceholder")}
                        value={agentSearchTerm}
                        onChange={(e) => setAgentSearchTerm(e.target.value)}
                      />
                      {agentSearchTerm && (
                        <InputGroupAddon align="inline-end">
                          <InputGroupButton
                            size="icon-xs"
                            aria-label={t("aiHub.modelHubTable.agents.clearSearch")}
                            onClick={() => setAgentSearchTerm("")}
                          >
                            <X />
                          </InputGroupButton>
                        </InputGroupAddon>
                      )}
                    </InputGroup>
                  </div>

                  {/* Agent Table */}
                  <DataTable
                    data={filteredAgentData}
                    paginationMode="client"
                    columns={agentColumns}
                    getRowId={(agent, index) => agent.agent_id || agent.name || String(index)}
                    sortingMode="client"
                    sorting={agentSorting}
                    onSortingChange={setAgentSorting}
                    isLoading={agentLoading}
                    loadingMessage={t("aiHub.modelHubTable.agents.loading")}
                    noDataMessage={
                      <HubEmptyState
                        title={
                          agentHubData?.length
                            ? t("aiHub.modelHubTable.agents.noMatchTitle")
                            : t("aiHub.modelHubTable.agents.noneTitle")
                        }
                        body={
                          agentHubData?.length
                            ? t("aiHub.modelHubTable.agents.noMatchBody")
                            : t("aiHub.modelHubTable.agents.noneBody")
                        }
                      />
                    }
                    size="compact"
                  />
                </Card>

                <div className="mt-4 text-center space-y-2">
                  <p className="text-sm text-muted-foreground">
                    {t("aiHub.modelHubTable.agents.showing", {
                      shown: filteredAgentData.length,
                      total: agentHubData?.length || 0,
                    })}
                  </p>
                </div>
              </TabsContent>

              {/* MCP Hub Tab */}
              <TabsContent value="mcp" keepMounted>
                <Card className="px-6">
                  {/* Header with Make Public Button */}
                  {publicPage == false && canModify && (
                    <div className="flex justify-end mb-4">
                      <Button
                        onClick={() => handleMakeMcpPublicPage()}
                        disabled={isMCPHubVisibilityDisabled(mcpLoading, mcpHubData)}
                      >
                        {t("aiHub.modelHubTable.mcp.manageVisibility")}
                      </Button>
                    </div>
                  )}

                  {/* MCP Server Table */}
                  <DataTable
                    data={mcpHubData || []}
                    paginationMode="client"
                    columns={mcpColumns}
                    getRowId={(server, index) => server.server_id || String(index)}
                    sortingMode="client"
                    sorting={mcpSorting}
                    onSortingChange={setMcpSorting}
                    isLoading={mcpLoading}
                    loadingMessage={t("aiHub.modelHubTable.mcp.loading")}
                    noDataMessage={
                      <HubEmptyState
                        title={t("aiHub.modelHubTable.mcp.noneTitle")}
                        body={t("aiHub.modelHubTable.mcp.noneBody")}
                      />
                    }
                    size="compact"
                  />
                </Card>

                <div className="mt-4 text-center space-y-2">
                  <p className="text-sm text-muted-foreground">
                    {t("aiHub.modelHubTable.mcp.showing", { count: mcpHubData?.length || 0 })}
                  </p>
                </div>
              </TabsContent>

              {/* Skill Hub Tab */}
              <TabsContent value="skills" keepMounted>
                {publicPage == false && canModify && (
                  <div className="flex justify-end mb-4">
                    <Button onClick={() => setIsMakeSkillPublicModalVisible(true)}>
                      {t("aiHub.modelHubTable.skills.makePublic")}
                    </Button>
                  </div>
                )}
                <SkillHubDashboard
                  skills={skillHubData}
                  isLoading={skillLoading}
                  isAdmin={canModify}
                  accessToken={accessToken}
                  publicPage={publicPage}
                  onPublishSuccess={async () => {
                    const response = await getClaudeCodePluginsList(accessToken || "", publicPage);
                    setSkillHubData(response.plugins);
                  }}
                />
              </TabsContent>
            </div>
          </Tabs>
        </div>
      ) : (
        <Card className="mx-auto max-w-xl mt-10 px-6">
          <p className="text-xl text-center mb-2 text-foreground">{t("aiHub.modelHubTable.publicNotEnabled.title")}</p>
          <p className="text-base text-center text-foreground">{t("aiHub.modelHubTable.publicNotEnabled.body")}</p>
        </Card>
      )}

      {/* Model Details Modal */}
      <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleCancel()}>
        <DialogContent
          className={`${isChat ? "chat-theme chat-model-hub-dialog " : ""}max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]`}
        >
          <DialogHeader>
            <DialogTitle>
              {selectedModel?.model_group || t("aiHub.modelHubTable.modelDialog.fallbackTitle")}
            </DialogTitle>
          </DialogHeader>
          {selectedModel && (
            <div className="space-y-6">
              {/* Model Overview */}
              <div>
                <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.modelDialog.overview")}</p>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.modelDialog.modelGroup")}</p>
                    <p>{selectedModel.model_group}</p>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.modelDialog.mode")}</p>
                    <p>{displayMode(selectedModel.mode) ?? t("aiHub.modelHubTable.modelDialog.notSpecified")}</p>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.modelDialog.providers")}</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {selectedModel.providers.map((provider) => (
                        <Badge key={provider} variant="secondary">
                          {provider}
                        </Badge>
                      ))}
                    </div>
                  </div>
                  {selectedModel.description && (
                    <div className="col-span-2">
                      <p className="font-medium">{t("aiHub.modelHubTable.modelDialog.description")}</p>
                      <p className="mt-1">{selectedModel.description}</p>
                    </div>
                  )}
                  <ModelCustomMetadata
                    customTags={selectedModel.custom_tags}
                    customMetadata={selectedModel.custom_metadata}
                  />
                </div>
              </div>

              {/* Token and Cost Information */}
              <div>
                <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.modelDialog.tokenCost")}</p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.modelDialog.maxInputTokens")}</p>
                    <p>
                      {selectedModel.max_input_tokens?.toLocaleString() ||
                        t("aiHub.modelHubTable.modelDialog.notSpecified")}
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.modelDialog.maxOutputTokens")}</p>
                    <p>
                      {selectedModel.max_output_tokens?.toLocaleString() ||
                        t("aiHub.modelHubTable.modelDialog.notSpecified")}
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.modelDialog.inputCost")}</p>
                    <p>
                      {selectedModel.input_cost_per_token
                        ? formatCost(selectedModel.input_cost_per_token)
                        : t("aiHub.modelHubTable.modelDialog.notSpecified")}
                    </p>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.modelDialog.outputCost")}</p>
                    <p>
                      {selectedModel.output_cost_per_token
                        ? formatCost(selectedModel.output_cost_per_token)
                        : t("aiHub.modelHubTable.modelDialog.notSpecified")}
                    </p>
                  </div>
                </div>
                <DeploymentPricingBreakdown
                  className="mt-4"
                  deployments={selectedModel.deployment_pricing}
                  heading={<p className="font-medium">{t("aiHub.modelHubTable.modelDialog.pricingByDeployment")}</p>}
                />
              </div>

              {/* Capabilities */}
              <div>
                <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.modelDialog.capabilities")}</p>
                <div className="flex flex-wrap gap-2">
                  {(() => {
                    const capabilities = getModelCapabilities(selectedModel);
                    if (capabilities.length === 0) {
                      return (
                        <p className="text-muted-foreground">{t("aiHub.modelHubTable.modelDialog.noCapabilities")}</p>
                      );
                    }

                    return capabilities.map((capability, index) => (
                      <Badge key={capability} variant="secondary">
                        {formatCapabilityName(capability)}
                      </Badge>
                    ));
                  })()}
                </div>
              </div>

              {/* Rate Limits */}
              {(selectedModel.tpm || selectedModel.rpm) && (
                <div>
                  <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.modelDialog.rateLimits")}</p>
                  <div className="grid grid-cols-2 gap-4">
                    {selectedModel.tpm && (
                      <div>
                        <p className="font-medium">{t("aiHub.modelHubTable.modelDialog.tpm")}</p>
                        <p>{selectedModel.tpm.toLocaleString()}</p>
                      </div>
                    )}
                    {selectedModel.rpm && (
                      <div>
                        <p className="font-medium">{t("aiHub.modelHubTable.modelDialog.rpm")}</p>
                        <p>{selectedModel.rpm.toLocaleString()}</p>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Supported OpenAI Parameters */}
              {selectedModel.supported_openai_params && (
                <div>
                  <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.modelDialog.openaiParams")}</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedModel.supported_openai_params.map((param) => (
                      <Badge key={param} variant="default">
                        {param}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              <div>
                <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.modelDialog.usageExample")}</p>
                {isDecisionMode(selectedModel.mode) && (
                  <p className="mb-2 text-sm text-muted-foreground">
                    <Trans i18nKey="aiHub.modelHubTable.modelDialog.decisionNote" components={{ code: <code /> }} />{" "}
                    <a href={DECISIONS_DOCS_URL} target="_blank" rel="noopener noreferrer" className="underline">
                      {t("aiHub.modelHubTable.modelDialog.howToCall")}
                    </a>
                    {canOpenPlayground && (
                      <>
                        {" · "}
                        <a href={uiHref(SYSTEM_ONE_PLAYGROUND_ROUTE)} className="underline">
                          {t("aiHub.modelHubTable.modelDialog.tryPlayground")}
                        </a>
                      </>
                    )}
                  </p>
                )}
                <SyntaxHighlighter language="python" className="text-sm" style={syntaxTheme}>
                  {modelUsageExample(
                    selectedModel.mode,
                    getGatewayBaseUrl(),
                    selectedModel.model_group,
                    getGlobalLitellmHeaderName(),
                  )}
                </SyntaxHighlighter>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Agent Details Modal */}
      <Dialog open={isAgentModalVisible} onOpenChange={(open) => !open && handleCancel()}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]">
          <DialogHeader>
            <DialogTitle>{selectedAgent?.name || t("aiHub.modelHubTable.agentDialog.fallbackTitle")}</DialogTitle>
          </DialogHeader>
          {selectedAgent && (
            <div className="space-y-6">
              {/* Agent Overview */}
              <div>
                <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.agentDialog.overview")}</p>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.agentDialog.name")}</p>
                    <p>{selectedAgent.name}</p>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.agentDialog.version")}</p>
                    <Badge variant="secondary">v{selectedAgent.version}</Badge>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.agentDialog.protocolVersion")}</p>
                    <p>{selectedAgent.protocolVersion}</p>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.agentDialog.url")}</p>
                    <div className="flex items-center space-x-2">
                      <p className="truncate min-w-0">{selectedAgent.url}</p>
                      <Copy
                        onClick={() => void copyToClipboard(selectedAgent.url)}
                        className="size-3.5 shrink-0 cursor-pointer text-muted-foreground hover:text-info"
                      />
                    </div>
                  </div>
                </div>
                <div>
                  <p className="font-medium">{t("aiHub.modelHubTable.agentDialog.description")}</p>
                  <p className="mt-1">{selectedAgent.description}</p>
                </div>
              </div>

              {/* Capabilities */}
              {selectedAgent.capabilities && Object.keys(selectedAgent.capabilities).length > 0 && (
                <div>
                  <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.agentDialog.capabilities")}</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(selectedAgent.capabilities)
                      .filter(([_, value]) => value === true)
                      .map(([key]) => (
                        <Badge key={key} variant="default">
                          {key}
                        </Badge>
                      ))}
                  </div>
                </div>
              )}

              {/* Input/Output Modes */}
              <div>
                <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.agentDialog.ioModes")}</p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.agentDialog.inputModes")}</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {selectedAgent.defaultInputModes?.map((mode) => (
                        <Badge key={mode} variant="secondary">
                          {mode}
                        </Badge>
                      )) || <p>{t("aiHub.modelHubTable.agentDialog.notSpecified")}</p>}
                    </div>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.agentDialog.outputModes")}</p>
                    <div className="flex flex-wrap gap-1 mt-1">
                      {selectedAgent.defaultOutputModes?.map((mode) => (
                        <Badge key={mode} variant="outline">
                          {mode}
                        </Badge>
                      )) || <p>{t("aiHub.modelHubTable.agentDialog.notSpecified")}</p>}
                    </div>
                  </div>
                </div>
              </div>

              {/* Skills */}
              {selectedAgent.skills && selectedAgent.skills.length > 0 && (
                <div>
                  <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.agentDialog.skills")}</p>
                  <div className="space-y-4">
                    {selectedAgent.skills.map((skill) => (
                      <div key={skill.id} className="border border-border rounded-sm p-4">
                        <div className="flex justify-between items-start mb-2">
                          <div>
                            <p className="font-medium text-base">{skill.name}</p>
                            <p className="text-xs text-muted-foreground">
                              {t("aiHub.modelHubTable.agentDialog.skillId", { id: skill.id })}
                            </p>
                          </div>
                          {skill.tags && skill.tags.length > 0 && (
                            <div className="flex flex-wrap gap-1">
                              {skill.tags.map((tag) => (
                                <Badge key={tag} variant="outline">
                                  {tag}
                                </Badge>
                              ))}
                            </div>
                          )}
                        </div>
                        <p className="text-sm mb-2">{skill.description}</p>
                        {skill.examples && skill.examples.length > 0 && (
                          <div>
                            <p className="text-xs font-medium text-foreground">
                              {t("aiHub.modelHubTable.agentDialog.examples")}
                            </p>
                            <div className="flex flex-wrap gap-1 mt-1">
                              {skill.examples.map((example, idx) => (
                                <Badge key={idx} variant="outline">
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
                  <p className="text-lg font-semibold mb-4">
                    {t("aiHub.modelHubTable.agentDialog.additionalFeatures")}
                  </p>
                  <Badge variant="default">{t("aiHub.modelHubTable.agentDialog.extendedCard")}</Badge>
                </div>
              )}
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* MCP Server Details Modal */}
      <Dialog open={isMcpModalVisible} onOpenChange={(open) => !open && handleCancel()}>
        <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]">
          <DialogHeader>
            <DialogTitle>
              {selectedMcpServer?.server_name || t("aiHub.modelHubTable.mcpDialog.fallbackTitle")}
            </DialogTitle>
          </DialogHeader>
          {selectedMcpServer && (
            <div className="space-y-6">
              {/* Server Overview */}
              <div>
                <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.mcpDialog.overview")}</p>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.mcpDialog.serverName")}</p>
                    <p>{selectedMcpServer.server_name}</p>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.mcpDialog.serverId")}</p>
                    <div className="flex items-center space-x-2">
                      <p className="text-xs truncate min-w-0">{selectedMcpServer.server_id}</p>
                      <Copy
                        onClick={() => void copyToClipboard(selectedMcpServer.server_id)}
                        className="size-3.5 shrink-0 cursor-pointer text-muted-foreground hover:text-info"
                      />
                    </div>
                  </div>
                  {selectedMcpServer.alias && (
                    <div>
                      <p className="font-medium">{t("aiHub.modelHubTable.mcpDialog.alias")}</p>
                      <p>{selectedMcpServer.alias}</p>
                    </div>
                  )}
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.mcpDialog.transport")}</p>
                    <Badge variant="secondary">{selectedMcpServer.transport}</Badge>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.mcpDialog.authType")}</p>
                    <Badge variant={selectedMcpServer.auth_type === "none" ? "outline" : "default"}>
                      {selectedMcpServer.auth_type}
                    </Badge>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.mcpDialog.status")}</p>
                    <Badge
                      variant={
                        selectedMcpServer.status === "active" || selectedMcpServer.status === "healthy"
                          ? "default"
                          : selectedMcpServer.status === "inactive" || selectedMcpServer.status === "unhealthy"
                            ? "destructive"
                            : "outline"
                      }
                    >
                      {selectedMcpServer.status || t("aiHub.modelHubTable.mcpDialog.statusUnknown")}
                    </Badge>
                  </div>
                </div>
                {selectedMcpServer.description && (
                  <div className="mt-2">
                    <p className="font-medium">{t("aiHub.modelHubTable.mcpDialog.description")}</p>
                    <p className="mt-1">{selectedMcpServer.description}</p>
                  </div>
                )}
              </div>

              {/* Connection Details */}
              <div>
                <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.mcpDialog.connectionDetails")}</p>
                <div className="space-y-2">
                  {selectedMcpServer.command && (
                    <div>
                      <p className="font-medium">{t("aiHub.modelHubTable.mcpDialog.command")}</p>
                      <p className="text-sm bg-muted p-2 rounded-sm mt-1 font-mono">{selectedMcpServer.command}</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Tools */}
              {selectedMcpServer.allowed_tools && selectedMcpServer.allowed_tools.length > 0 && (
                <div>
                  <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.mcpDialog.allowedTools")}</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedMcpServer.allowed_tools.map((tool, idx) => (
                      <Badge key={idx} variant="outline">
                        {tool}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Teams */}
              {selectedMcpServer.teams && selectedMcpServer.teams.length > 0 && (
                <div>
                  <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.mcpDialog.teams")}</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedMcpServer.teams.map((team, idx) => (
                      <Badge key={idx} variant="secondary">
                        {team}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Access Groups */}
              {selectedMcpServer.mcp_access_groups && selectedMcpServer.mcp_access_groups.length > 0 && (
                <div>
                  <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.mcpDialog.accessGroups")}</p>
                  <div className="flex flex-wrap gap-2">
                    {selectedMcpServer.mcp_access_groups.map((group, idx) => (
                      <Badge key={idx} variant="default">
                        {group}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Metadata */}
              <div>
                <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.mcpDialog.metadata")}</p>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.mcpDialog.createdBy")}</p>
                    <p>{selectedMcpServer.created_by}</p>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.mcpDialog.updatedBy")}</p>
                    <p>{selectedMcpServer.updated_by}</p>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.mcpDialog.createdAt")}</p>
                    <p className="text-sm">{new Date(selectedMcpServer.created_at).toLocaleString()}</p>
                  </div>
                  <div>
                    <p className="font-medium">{t("aiHub.modelHubTable.mcpDialog.updatedAt")}</p>
                    <p className="text-sm">{new Date(selectedMcpServer.updated_at).toLocaleString()}</p>
                  </div>
                  {selectedMcpServer.last_health_check && (
                    <div>
                      <p className="font-medium">{t("aiHub.modelHubTable.mcpDialog.lastHealthCheck")}</p>
                      <p className="text-sm">{new Date(selectedMcpServer.last_health_check).toLocaleString()}</p>
                    </div>
                  )}
                </div>
                {selectedMcpServer.health_check_error && (
                  <div className="mt-2 p-2 bg-destructive/10 rounded-sm">
                    <p className="font-medium text-destructive">
                      {t("aiHub.modelHubTable.mcpDialog.healthCheckError")}
                    </p>
                    <p className="text-sm text-destructive mt-1">{selectedMcpServer.health_check_error}</p>
                  </div>
                )}
              </div>

              {/* Usage Example */}
              <div>
                <p className="text-lg font-semibold mb-4">{t("aiHub.modelHubTable.mcpDialog.usageExample")}</p>
                <SyntaxHighlighter language="python" className="text-sm" style={syntaxTheme}>
                  {`from fastmcp import Client
import asyncio

# Standard MCP configuration
config = {
    "mcpServers": {
        "${selectedMcpServer.server_name}": {
            "url": "${getProxyBaseUrl()}/${selectedMcpServer.server_name}/mcp",
            "headers": {
                "x-litellm-api-key": "Bearer <your-master-key>"
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
