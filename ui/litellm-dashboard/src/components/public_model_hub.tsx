import { ThemeProvider } from "@/contexts/ThemeContext";
import { ExternalLinkIcon, SearchIcon } from "@heroicons/react/outline";
import { SortingState } from "@tanstack/react-table";
import { Copy, Inbox, Info } from "lucide-react";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Trans, useTranslation } from "react-i18next";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import {
  Combobox,
  ComboboxChip,
  ComboboxChips,
  ComboboxChipsInput,
  ComboboxContent,
  ComboboxEmpty,
  ComboboxItem,
  ComboboxList,
  ComboboxValue,
  useComboboxAnchor,
} from "@/components/ui/combobox";
import { Dialog, DialogContent, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import { MultiSelect } from "./shared/MultiSelect";
import { featureLabel } from "./publicModelHub/publicModelHubFilters";
import { usePublicModelHubFacets } from "./publicModelHub/usePublicModelHubFacets";
import { usePublicModelHubList } from "./publicModelHub/usePublicModelHubList";
import { DataTable } from "./shared/DataTable";
import { toast } from "@/lib/toast";
import Navbar from "./navbar";
import {
  agentHubPublicModelsCall,
  skillHubPublicCall,
  getGlobalLitellmHeaderName,
  getProxyBaseUrl,
  getPublicModelHubInfo,
  getUiConfig,
  mcpHubPublicServersCall,
} from "./networking";
import { Plugin } from "./claude_code_plugins/types";
import SkillHubDashboard from "./AIHub/SkillHubDashboard";
import {
  AgentCard,
  MCPServerData,
  ModelGroupInfo,
  getPublicAgentHubColumns,
  getPublicMCPHubColumns,
  getPublicModelHubColumns,
} from "./PublicModelHubTableColumns";
import { generateCodeSnippet } from "@/components/chat_ui/CodeSnippets";
import { getEndpointType } from "@/components/chat_ui/mode_endpoint_mapping";
import { MessageType } from "@/components/chat_ui/types";
import { getProviderLogoAndName } from "./provider_info_helpers";
import { displayMode, isDecisionMode } from "@/lib/decisionModels";
import { modelUsageExample } from "@/components/AIHub/modelUsageExample";
import { filterBySearchTerm, rankBySearchRelevance } from "@/utils/searchUtils";
import { ModelCustomMetadata } from "@/components/AIHub/ModelCustomMetadata";
import { DeploymentPricingBreakdown } from "@/components/molecules/models/DeploymentPricingBreakdown";
import ModelHubProviders, { formatModelModeLabel } from "@/components/AIHub/ModelHubProviders";

interface PublicModelHubProps {
  accessToken?: string | null;
  isEmbedded?: boolean; // When true, hides navbar and adjusts layout for embedding in dashboard
  variant?: "dashboard" | "chat";
}

const usageExample = (model: ModelGroupInfo): string =>
  isDecisionMode(model.mode)
    ? modelUsageExample(model.mode, getProxyBaseUrl(), model.model_group, getGlobalLitellmHeaderName())
    : generateCodeSnippet({
        apiKeySource: "custom",
        accessToken: null,
        apiKey: "your_api_key",
        inputMessage: "Hello, how are you?",
        chatHistory: [{ role: "user", content: "Hello, how are you?", isImage: false } as MessageType],
        selectedTags: [],
        selectedVectorStores: [],
        selectedGuardrails: [],
        selectedPolicies: [],
        selectedMCPServers: [],
        endpointType: getEndpointType(model.mode || "chat"),
        selectedModel: model.model_group,
        selectedSdk: "openai",
      });

function PublicHubEmptyState({ title, body }: { title: string; body: string }) {
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

const PublicModelHub: React.FC<PublicModelHubProps> = ({ accessToken, isEmbedded = false, variant = "dashboard" }) => {
  const { t } = useTranslation();
  const isChat = variant === "chat";
  const portalClassName = isChat ? "chat-theme" : undefined;
  const dashboardRootClassName = isEmbedded ? "w-full" : "min-h-screen bg-card";
  const dashboardContentClassName = isEmbedded ? "w-full p-6" : "w-full px-8 py-12";
  const anchor = useComboboxAnchor();
  const [proxyConfigured, setProxyConfigured] = useState<boolean>(false);
  const [agentHubData, setAgentHubData] = useState<AgentCard[] | null>(null);
  const [mcpHubData, setMcpHubData] = useState<MCPServerData[] | null>(null);
  const [pageTitle, setPageTitle] = useState<string>("LiteLLM Gateway");
  const [customDocsDescription, setCustomDocsDescription] = useState<string | null>(null);
  const [litellmVersion, setLitellmVersion] = useState<string>("");
  const [usefulLinks, setUsefulLinks] = useState<Record<string, string | { url: string; index: number }>>({});
  const [agentLoading, setAgentLoading] = useState<boolean>(true);
  const [mcpLoading, setMcpLoading] = useState<boolean>(true);
  const [agentSearchTerm, setAgentSearchTerm] = useState<string>("");
  const [mcpSearchTerm, setMcpSearchTerm] = useState<string>("");
  const [selectedAgentSkills, setSelectedAgentSkills] = useState<string[]>([]);
  const [selectedMcpTransports, setSelectedMcpTransports] = useState<string[]>([]);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [isAgentModalVisible, setIsAgentModalVisible] = useState(false);
  const [isMcpModalVisible, setIsMcpModalVisible] = useState(false);
  const [selectedModel, setSelectedModel] = useState<null | ModelGroupInfo>(null);
  const [selectedAgent, setSelectedAgent] = useState<null | AgentCard>(null);
  const [selectedMcpServer, setSelectedMcpServer] = useState<null | MCPServerData>(null);
  const [activeTab, setActiveTab] = useState<string>("models");
  const [skillHubData, setSkillHubData] = useState<Plugin[]>([]);
  const [skillLoading, setSkillLoading] = useState<boolean>(false);

  useEffect(() => {
    const initializeAndFetch = async () => {
      // Initialize proxyBaseUrl first to ensure it includes the server root path
      try {
        await getUiConfig();
      } catch (error) {
        console.error("Failed to get UI config:", error);
        // Continue anyway - might work with default proxyBaseUrl
      }
      setProxyConfigured(true);
      if (isChat) return;

      const fetchAgentData = async () => {
        try {
          setAgentLoading(true);
          const _agentHubData = await agentHubPublicModelsCall();
          setAgentHubData(Array.isArray(_agentHubData) ? _agentHubData : []);
        } catch (error) {
          console.error("There was an error fetching the public agent data", error);
        } finally {
          setAgentLoading(false);
        }
      };

      const fetchMcpData = async () => {
        try {
          setMcpLoading(true);
          const _mcpHubData = await mcpHubPublicServersCall();
          setMcpHubData(Array.isArray(_mcpHubData) ? _mcpHubData : []);
        } catch (error) {
          console.error("There was an error fetching the public MCP server data", error);
        } finally {
          setMcpLoading(false);
        }
      };

      const fetchPublicModelHubInfo = async () => {
        const publicModelHubInfo = await getPublicModelHubInfo();
        setPageTitle(publicModelHubInfo.docs_title);
        setCustomDocsDescription(publicModelHubInfo.custom_docs_description);
        setLitellmVersion(publicModelHubInfo.litellm_version);
        setUsefulLinks(publicModelHubInfo.useful_links || {});
      };

      const fetchSkillData = async () => {
        try {
          setSkillLoading(true);
          const response = await skillHubPublicCall();
          setSkillHubData(response.plugins ?? []);
        } catch (error) {
          console.error("There was an error fetching the public skill data", error);
        } finally {
          setSkillLoading(false);
        }
      };

      fetchPublicModelHubInfo();

      fetchAgentData();
      fetchMcpData();
      fetchSkillData();
    };

    initializeAndFetch();
  }, [isChat]);

  const getUniqueAgentSkills = (data: AgentCard[]) => {
    const skills = new Set<string>();
    data.forEach((agent) => {
      agent.skills?.forEach((skill) => {
        skill.tags?.forEach((tag) => skills.add(tag));
      });
    });
    return Array.from(skills).sort();
  };

  const getUniqueMcpTransports = (data: MCPServerData[]) => {
    const transports = new Set<string>();
    data.forEach((server) => {
      if (server.transport) transports.add(server.transport);
    });
    return Array.from(transports).sort();
  };

  const filteredAgentData = useMemo(() => {
    if (!agentHubData || !Array.isArray(agentHubData)) return [];

    const searchResults = rankBySearchRelevance(
      filterBySearchTerm(agentHubData, agentSearchTerm, (agent) => [agent.name, agent.description]),
      agentSearchTerm,
      (agent) => agent.name,
    );

    // Apply skill filters
    return searchResults.filter((agent) => {
      const matchesSkill =
        selectedAgentSkills.length === 0 ||
        agent.skills?.some((skill) => skill.tags?.some((tag) => selectedAgentSkills.includes(tag)));

      return matchesSkill;
    });
  }, [agentHubData, agentSearchTerm, selectedAgentSkills]);

  const filteredMcpData = useMemo(() => {
    if (!mcpHubData || !Array.isArray(mcpHubData)) return [];

    const searchResults = rankBySearchRelevance(
      filterBySearchTerm(mcpHubData, mcpSearchTerm, (server) => [server.server_name, server.mcp_info?.description]),
      mcpSearchTerm,
      (server) => server.server_name,
    );

    // Apply transport filters
    return searchResults.filter((server) => {
      const matchesTransport = selectedMcpTransports.length === 0 || selectedMcpTransports.includes(server.transport);

      return matchesTransport;
    });
  }, [mcpHubData, mcpSearchTerm, selectedMcpTransports]);

  const showModal = useCallback((model: ModelGroupInfo) => {
    setSelectedModel(model);
    setIsModalVisible(true);
  }, []);

  const handleModalCancel = () => {
    setIsModalVisible(false);
    setSelectedModel(null);
  };

  const showAgentModal = useCallback((agent: AgentCard) => {
    setSelectedAgent(agent);
    setIsAgentModalVisible(true);
  }, []);

  const handleAgentModalCancel = () => {
    setIsAgentModalVisible(false);
    setSelectedAgent(null);
  };

  const showMcpModal = useCallback((server: MCPServerData) => {
    setSelectedMcpServer(server);
    setIsMcpModalVisible(true);
  }, []);

  const handleMcpModalCancel = () => {
    setIsMcpModalVisible(false);
    setSelectedMcpServer(null);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    toast.success(t("aiHub.publicModelHub.copiedToClipboard"));
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
    return `$${(cost * 1_000_000).toFixed(4)}`;
  };

  const models = usePublicModelHubList(proxyConfigured);
  const modelFacets = usePublicModelHubFacets(proxyConfigured);
  const modeOptions = useMemo(
    () =>
      [...new Set(modelFacets.modes.flatMap((mode) => displayMode(mode) ?? []))].map((mode) => ({
        label: isChat ? formatModelModeLabel(mode) : mode,
        value: mode,
      })),
    [modelFacets, isChat],
  );
  const featureOptions = useMemo(
    () => modelFacets.features.map((feature) => ({ label: featureLabel(feature), value: feature })),
    [modelFacets],
  );
  const serviceStatus = models.error
    ? t("aiHub.publicModelHub.serviceUnavailable")
    : t("aiHub.publicModelHub.serviceAlive");
  const [agentSorting, setAgentSorting] = useState<SortingState>([{ id: "name", desc: false }]);
  const [mcpSorting, setMcpSorting] = useState<SortingState>([{ id: "server_name", desc: false }]);

  const modelColumns = useMemo(
    () => getPublicModelHubColumns({ onModelClick: showModal, showProviderLogos: isChat }),
    [showModal, isChat],
  );
  const agentColumns = useMemo(() => getPublicAgentHubColumns({ onAgentClick: showAgentModal }), [showAgentModal]);
  const mcpColumns = useMemo(() => getPublicMCPHubColumns({ onServerClick: showMcpModal }), [showMcpModal]);

  const showUsefulLinks = !isChat && Object.keys(usefulLinks).length > 0;
  const hasAgents = Array.isArray(agentHubData) && agentHubData.length > 0;
  const hasMcpServers = Array.isArray(mcpHubData) && mcpHubData.length > 0;

  const agentSkillOptions = useMemo(
    () =>
      Array.isArray(agentHubData)
        ? getUniqueAgentSkills(agentHubData).map((skill) => ({ label: skill, value: skill }))
        : [],
    [agentHubData],
  );
  const mcpTransportOptions = useMemo(
    () =>
      Array.isArray(mcpHubData)
        ? getUniqueMcpTransports(mcpHubData).map((transport) => ({ label: transport, value: transport }))
        : [],
    [mcpHubData],
  );

  return (
    <ThemeProvider accessToken={accessToken}>
      <TooltipProvider>
        <div className={isChat ? "chat-model-hub-catalog w-full" : dashboardRootClassName}>
          {/* Navigation - only show when not embedded */}
          {!isChat && !isEmbedded && <Navbar accessToken={accessToken || null} isPublicPage={true} />}

          <div className={isChat ? "chat-model-hub-content w-full" : dashboardContentClassName}>
            {/* Embedded Explainer - only shown when embedded in dashboard */}
            {!isChat && isEmbedded && (
              <div className="mb-6 p-4 bg-info/10 border border-info/20 rounded-lg">
                <p className="text-sm text-foreground">{t("aiHub.publicModelHub.embeddedExplainer")}</p>
              </div>
            )}

            {/* About Section - only shown when not embedded */}
            {!isChat && !isEmbedded && (
              <Card className="mb-10 p-8 bg-card border border-border rounded-lg shadow-xs">
                <h2 className="text-2xl font-semibold mb-6 text-foreground">{t("aiHub.publicModelHub.about.title")}</h2>
                <p className="text-foreground mb-6 text-base leading-relaxed">
                  {customDocsDescription ? customDocsDescription : t("aiHub.publicModelHub.about.defaultDescription")}
                </p>
                <div className="flex items-center space-x-3 text-sm text-muted-foreground">
                  <span className="flex items-center">
                    <span className="w-4 h-4 mr-2">🔧</span>
                    {t("aiHub.publicModelHub.about.builtWith", { version: litellmVersion })}
                  </span>
                </div>
              </Card>
            )}

            {/* Useful Links - only shown when not embedded */}
            {showUsefulLinks && (
              <Card className="mb-10 p-8 bg-card border border-border rounded-lg shadow-xs">
                <h2 className="text-2xl font-semibold mb-6 text-foreground">
                  {t("aiHub.publicModelHub.usefulLinks.title")}
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {Object.entries(usefulLinks || {})
                    .map(([title, value]) => {
                      // Handle both old format (string) and new format ({url, index})
                      const url = typeof value === "string" ? value : value.url;
                      const index = typeof value === "string" ? 0 : value.index ?? 0;
                      return { title, url, index };
                    })
                    .sort((a, b) => a.index - b.index)
                    .map(({ title, url }) => (
                      <button
                        key={title}
                        onClick={() => window.open(url, "_blank")}
                        className="flex min-w-0 items-center space-x-3 text-info transition-colors p-3 rounded-lg hover:bg-info/10 border border-border"
                      >
                        <ExternalLinkIcon className="w-4 h-4 shrink-0" />
                        <p className="text-sm font-medium break-words">{title}</p>
                      </button>
                    ))}
                </div>
              </Card>
            )}

            {/* Health and Endpoint Status - only shown when not embedded */}
            {!isChat && !isEmbedded && (
              <Card className="mb-10 p-8 bg-card border border-border rounded-lg shadow-xs">
                <h2 className="text-2xl font-semibold mb-6 text-foreground">
                  {t("aiHub.publicModelHub.health.title")}
                </h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <p className="text-success font-medium text-sm">
                    {t("aiHub.publicModelHub.health.serviceStatus", { status: serviceStatus })}
                  </p>
                </div>
              </Card>
            )}

            {/* Tabs for Models and Agents */}
            <Card
              className={
                isChat
                  ? "chat-model-hub-card border-0 bg-transparent p-0 shadow-none"
                  : "p-8 bg-card border border-border rounded-lg shadow-xs"
              }
            >
              <Tabs value={isChat ? "models" : activeTab} onValueChange={setActiveTab} className="public-hub-tabs">
                {!isChat && (
                  <TabsList>
                    <TabsTrigger value="models">{t("aiHub.publicModelHub.tabs.models")}</TabsTrigger>
                    {hasAgents && <TabsTrigger value="agents">{t("aiHub.publicModelHub.tabs.agents")}</TabsTrigger>}
                    {hasMcpServers && <TabsTrigger value="mcp">{t("aiHub.publicModelHub.tabs.mcp")}</TabsTrigger>}
                    <TabsTrigger value="skills">{t("aiHub.publicModelHub.tabs.skills")}</TabsTrigger>
                  </TabsList>
                )}

                {/* Models Tab */}
                <TabsContent value="models" className="model-hub-models">
                  {!isChat && (
                    <div className="model-hub-heading flex justify-between items-center mb-8">
                      <h2 className="text-2xl font-semibold text-foreground">
                        {t("aiHub.publicModelHub.models.title")}
                      </h2>
                    </div>
                  )}

                  {isChat && (
                    <div className="chat-model-section-heading">
                      <h2>{t("aiHub.publicModelHub.models.chatTitle")}</h2>
                      <span
                        className="chat-model-section-count"
                        aria-label={t("aiHub.publicModelHub.models.chatCount", { rowCount: models.rowCount })}
                      >
                        {models.rowCount.toLocaleString()}
                      </span>
                    </div>
                  )}

                  {/* Filters */}
                  <div
                    className={
                      isChat
                        ? "chat-model-hub-filters chat-model-filter-bar"
                        : "chat-model-hub-filters grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8 p-6 bg-muted rounded-lg border border-border"
                    }
                  >
                    <div className={isChat ? "chat-model-filter-search" : undefined}>
                      {!isChat && (
                        <div className="flex items-center space-x-2 mb-3">
                          <p className="text-sm font-medium text-foreground">
                            {t("aiHub.publicModelHub.models.searchLabel")}
                          </p>
                          <Tooltip>
                            <TooltipTrigger render={<Info className="w-4 h-4 text-muted-foreground cursor-help" />} />
                            <TooltipContent side="top" className={portalClassName}>
                              {t("aiHub.publicModelHub.models.searchTooltip")}
                            </TooltipContent>
                          </Tooltip>
                        </div>
                      )}
                      <div className="relative">
                        <SearchIcon className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 transform -translate-y-1/2" />
                        {isChat ? (
                          <Input
                            type="text"
                            placeholder={t("aiHub.publicModelHub.models.chatSearchPlaceholder")}
                            aria-label={t("aiHub.publicModelHub.models.chatSearchAriaLabel")}
                            value={models.searchValue}
                            onChange={(e) => models.onSearchChange(e.target.value)}
                            className="pl-9"
                          />
                        ) : (
                          <input
                            type="text"
                            placeholder={t("aiHub.publicModelHub.models.searchPlaceholder")}
                            aria-label={t("aiHub.publicModelHub.models.searchAriaLabel")}
                            value={models.searchValue}
                            onChange={(e) => models.onSearchChange(e.target.value)}
                            className="border border-border rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-hidden focus:ring-2 focus:ring-ring focus:border-transparent bg-card"
                          />
                        )}
                      </div>
                    </div>
                    {isChat && (
                      <ModelHubProviders
                        providers={modelFacets.providers}
                        selectedProviders={models.providerValues}
                        onSelect={(provider) => models.onProvidersChange(provider ? [provider] : [])}
                        loading={models.isLoading && modelFacets.providers.length === 0}
                      />
                    )}
                    {!isChat && (
                      <div className="min-w-0">
                        <p className="text-sm font-medium mb-3 text-foreground">
                          {t("aiHub.publicModelHub.models.providerLabel")}
                        </p>
                        <Combobox
                          multiple
                          items={modelFacets.providers}
                          value={models.providerValues}
                          onValueChange={models.onProvidersChange}
                        >
                          <ComboboxChips render={<div ref={anchor} />} className="min-h-8 w-full py-1 text-sm">
                            <ComboboxValue>
                              {(values: string[]) =>
                                values.map((provider) => (
                                  <ComboboxChip key={provider} aria-label={provider}>
                                    {provider}
                                  </ComboboxChip>
                                ))
                              }
                            </ComboboxValue>
                            <ComboboxChipsInput
                              placeholder={t("aiHub.publicModelHub.models.providersPlaceholder")}
                              aria-label={t("aiHub.publicModelHub.models.providersAriaLabel")}
                              className="min-w-24"
                            />
                          </ComboboxChips>
                          <ComboboxContent anchor={anchor} className={portalClassName}>
                            <ComboboxEmpty>{t("aiHub.publicModelHub.models.noProvidersFound")}</ComboboxEmpty>
                            <ComboboxList>
                              {(provider: string) => {
                                const { logo } = getProviderLogoAndName(provider);
                                return (
                                  <ComboboxItem key={provider} value={provider}>
                                    <span className="flex min-w-0 items-center space-x-2">
                                      {logo && (
                                        <img
                                          src={logo}
                                          alt={provider}
                                          className="w-5 h-5 shrink-0 object-contain"
                                          onError={(e) => {
                                            (e.target as HTMLImageElement).style.display = "none";
                                          }}
                                        />
                                      )}
                                      <span className="capitalize break-words">{provider}</span>
                                    </span>
                                  </ComboboxItem>
                                );
                              }}
                            </ComboboxList>
                          </ComboboxContent>
                        </Combobox>
                      </div>
                    )}
                    <div className={isChat ? "chat-model-filter-select min-w-0" : "min-w-0"}>
                      {!isChat && (
                        <p className="text-sm font-medium mb-3 text-foreground">
                          {t("aiHub.publicModelHub.models.modeLabel")}
                        </p>
                      )}
                      <MultiSelect
                        options={modeOptions}
                        contentClassName={portalClassName}
                        value={models.modeValues}
                        onValueChange={models.onModesChange}
                        placeholder={
                          isChat
                            ? t("aiHub.publicModelHub.models.chatModePlaceholder")
                            : t("aiHub.publicModelHub.models.modePlaceholder")
                        }
                        className="w-full"
                      />
                    </div>
                    <div className={isChat ? "chat-model-filter-select min-w-0" : "min-w-0"}>
                      {!isChat && (
                        <p className="text-sm font-medium mb-3 text-foreground">
                          {t("aiHub.publicModelHub.models.featuresLabel")}
                        </p>
                      )}
                      <MultiSelect
                        options={featureOptions}
                        contentClassName={portalClassName}
                        value={models.featureValues}
                        onValueChange={models.onFeaturesChange}
                        placeholder={
                          isChat
                            ? t("aiHub.publicModelHub.models.chatFeaturesPlaceholder")
                            : t("aiHub.publicModelHub.models.featuresPlaceholder")
                        }
                        className="w-full"
                      />
                    </div>
                  </div>

                  <div className="model-hub-table">
                    <DataTable
                      data={models.rows}
                      columns={modelColumns}
                      getRowId={(model, index) => model.model_group || String(index)}
                      sortingMode="server"
                      sorting={models.sorting}
                      onSortingChange={models.onSortingChange}
                      paginationMode="server"
                      pagination={models.pagination}
                      onPaginationChange={models.onPaginationChange}
                      rowCount={models.rowCount}
                      isLoading={models.isLoading}
                      loadingMessage={t("aiHub.publicModelHub.models.loading")}
                      noDataMessage={
                        <PublicHubEmptyState
                          title={
                            models.hasActiveQuery
                              ? t("aiHub.publicModelHub.models.noMatchTitle")
                              : t("aiHub.publicModelHub.models.emptyTitle")
                          }
                          body={
                            models.hasActiveQuery
                              ? t("aiHub.publicModelHub.models.noMatchBody")
                              : t("aiHub.publicModelHub.models.emptyBody")
                          }
                        />
                      }
                      size="compact"
                    />
                  </div>
                </TabsContent>

                {/* Agents Tab */}
                {!isChat && hasAgents && (
                  <TabsContent value="agents">
                    <div className="flex justify-between items-center mb-8">
                      <h2 className="text-2xl font-semibold text-foreground">
                        {t("aiHub.publicModelHub.agents.title")}
                      </h2>
                    </div>

                    {/* Filters */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-muted rounded-lg border border-border">
                      <div>
                        <div className="flex items-center space-x-2 mb-3">
                          <p className="text-sm font-medium text-foreground">
                            {t("aiHub.publicModelHub.agents.searchLabel")}
                          </p>
                          <Tooltip>
                            <TooltipTrigger render={<Info className="w-4 h-4 text-muted-foreground cursor-help" />} />
                            <TooltipContent side="top" className={portalClassName}>
                              {t("aiHub.publicModelHub.agents.searchTooltip")}
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <div className="relative">
                          <SearchIcon className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 transform -translate-y-1/2" />
                          <input
                            type="text"
                            placeholder={t("aiHub.publicModelHub.agents.searchPlaceholder")}
                            value={agentSearchTerm}
                            onChange={(e) => setAgentSearchTerm(e.target.value)}
                            className="border border-border rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-hidden focus:ring-2 focus:ring-ring focus:border-transparent bg-card"
                          />
                        </div>
                      </div>
                      <div className="min-w-0">
                        <p className="text-sm font-medium mb-3 text-foreground">
                          {t("aiHub.publicModelHub.agents.skillsLabel")}
                        </p>
                        <MultiSelect
                          options={agentSkillOptions}
                          value={selectedAgentSkills}
                          onValueChange={setSelectedAgentSkills}
                          placeholder={t("aiHub.publicModelHub.agents.skillsPlaceholder")}
                          className="w-full"
                        />
                      </div>
                    </div>

                    <DataTable
                      data={filteredAgentData}
                      paginationMode="client"
                      columns={agentColumns}
                      getRowId={(agent, index) => agent.name || String(index)}
                      sortingMode="client"
                      sorting={agentSorting}
                      onSortingChange={setAgentSorting}
                      isLoading={agentLoading}
                      loadingMessage={t("aiHub.publicModelHub.agents.loading")}
                      noDataMessage={
                        <PublicHubEmptyState
                          title={t("aiHub.publicModelHub.agents.noMatchTitle")}
                          body={t("aiHub.publicModelHub.agents.noMatchBody")}
                        />
                      }
                      size="compact"
                    />

                    <div className="mt-8 text-center">
                      <p className="text-sm text-muted-foreground">
                        {t("aiHub.publicModelHub.agents.showing", {
                          shown: filteredAgentData.length,
                          total: agentHubData?.length || 0,
                        })}
                      </p>
                    </div>
                  </TabsContent>
                )}

                {/* MCP Servers Tab */}
                {!isChat && hasMcpServers && (
                  <TabsContent value="mcp">
                    <div className="flex justify-between items-center mb-8">
                      <h2 className="text-2xl font-semibold text-foreground">{t("aiHub.publicModelHub.mcp.title")}</h2>
                    </div>

                    {/* Filters */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-muted rounded-lg border border-border">
                      <div>
                        <div className="flex items-center space-x-2 mb-3">
                          <p className="text-sm font-medium text-foreground">
                            {t("aiHub.publicModelHub.mcp.searchLabel")}
                          </p>
                          <Tooltip>
                            <TooltipTrigger render={<Info className="w-4 h-4 text-muted-foreground cursor-help" />} />
                            <TooltipContent side="top" className={portalClassName}>
                              {t("aiHub.publicModelHub.mcp.searchTooltip")}
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <div className="relative">
                          <SearchIcon className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 transform -translate-y-1/2" />
                          <input
                            type="text"
                            placeholder={t("aiHub.publicModelHub.mcp.searchPlaceholder")}
                            value={mcpSearchTerm}
                            onChange={(e) => setMcpSearchTerm(e.target.value)}
                            className="border border-border rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-hidden focus:ring-2 focus:ring-ring focus:border-transparent bg-card"
                          />
                        </div>
                      </div>
                      <div className="min-w-0">
                        <p className="text-sm font-medium mb-3 text-foreground">
                          {t("aiHub.publicModelHub.mcp.transportLabel")}
                        </p>
                        <MultiSelect
                          options={mcpTransportOptions}
                          value={selectedMcpTransports}
                          onValueChange={setSelectedMcpTransports}
                          placeholder={t("aiHub.publicModelHub.mcp.transportPlaceholder")}
                          className="w-full"
                        />
                      </div>
                    </div>

                    <DataTable
                      data={filteredMcpData}
                      paginationMode="client"
                      columns={mcpColumns}
                      getRowId={(server, index) => server.server_id || String(index)}
                      sortingMode="client"
                      sorting={mcpSorting}
                      onSortingChange={setMcpSorting}
                      isLoading={mcpLoading}
                      loadingMessage={t("aiHub.publicModelHub.mcp.loading")}
                      noDataMessage={
                        <PublicHubEmptyState
                          title={t("aiHub.publicModelHub.mcp.noMatchTitle")}
                          body={t("aiHub.publicModelHub.mcp.noMatchBody")}
                        />
                      }
                      size="compact"
                    />

                    <div className="mt-8 text-center">
                      <p className="text-sm text-muted-foreground">
                        {t("aiHub.publicModelHub.mcp.showing", {
                          shown: filteredMcpData.length,
                          total: mcpHubData?.length || 0,
                        })}
                      </p>
                    </div>
                  </TabsContent>
                )}

                {/* Skill Hub Tab */}
                {!isChat && (
                  <TabsContent value="skills">
                    <SkillHubDashboard skills={skillHubData} isLoading={skillLoading} publicPage={true} />
                  </TabsContent>
                )}
              </Tabs>
            </Card>
          </div>

          {/* Model Details Modal */}
          <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleModalCancel()}>
            <DialogContent
              className={`${isChat ? "chat-theme chat-model-hub-dialog " : ""}max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]`}
            >
              <DialogHeader>
                <DialogTitle className="flex min-w-0 items-center space-x-2">
                  <span className="break-words">
                    {selectedModel?.model_group || t("aiHub.publicModelHub.modelDialog.fallbackTitle")}
                  </span>
                  {selectedModel && (
                    <Tooltip>
                      <TooltipTrigger
                        render={
                          <Copy
                            onClick={() => copyToClipboard(selectedModel.model_group)}
                            className="cursor-pointer text-muted-foreground hover:text-info w-4 h-4 shrink-0"
                          />
                        }
                      />
                      <TooltipContent className={portalClassName}>
                        {t("aiHub.publicModelHub.modelDialog.copyName")}
                      </TooltipContent>
                    </Tooltip>
                  )}
                </DialogTitle>
              </DialogHeader>
              {selectedModel && (
                <div className="space-y-6">
                  {/* Model Overview */}
                  <div>
                    <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.modelDialog.overview")}</p>
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.modelDialog.modelName")}</p>
                        <p>{selectedModel.model_group}</p>
                      </div>
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.modelDialog.mode")}</p>
                        <p>{displayMode(selectedModel.mode) ?? t("aiHub.publicModelHub.modelDialog.notSpecified")}</p>
                      </div>
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.modelDialog.providers")}</p>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {(selectedModel.providers ?? []).map((provider) => {
                            const { logo } = getProviderLogoAndName(provider);
                            return (
                              <Badge key={provider} variant="secondary" className="min-w-0">
                                <div className="flex items-center space-x-1">
                                  {logo && (
                                    <img
                                      src={logo}
                                      alt={provider}
                                      className="w-3 h-3 shrink-0 object-contain"
                                      onError={(e) => {
                                        (e.target as HTMLImageElement).style.display = "none";
                                      }}
                                    />
                                  )}
                                  <span className="capitalize">{provider}</span>
                                </div>
                              </Badge>
                            );
                          })}
                        </div>
                      </div>
                      {selectedModel.description && (
                        <div className="col-span-2">
                          <p className="font-medium">{t("aiHub.publicModelHub.modelDialog.description")}</p>
                          <p className="mt-1">{selectedModel.description}</p>
                        </div>
                      )}
                      <ModelCustomMetadata
                        customTags={selectedModel.custom_tags}
                        customMetadata={selectedModel.custom_metadata}
                      />
                    </div>

                    {/* Wildcard Routing Note */}
                    {selectedModel.model_group.includes("*") && (
                      <div className="bg-info/10 border border-info/20 rounded-lg p-4 mb-4">
                        <div className="flex items-start space-x-2">
                          <Info className="w-4 h-4 text-info mt-0.5 shrink-0" />
                          <div>
                            <p className="font-medium text-info mb-2">
                              {t("aiHub.publicModelHub.modelDialog.wildcardTitle")}
                            </p>
                            <p className="text-sm text-info mb-2">
                              <Trans
                                i18nKey="aiHub.publicModelHub.modelDialog.wildcardIntro"
                                components={{ code: <code className="bg-info/15 px-1 py-0.5 rounded-sm text-xs" /> }}
                              />
                            </p>
                            <p className="text-sm text-info">
                              <Trans
                                i18nKey="aiHub.publicModelHub.modelDialog.wildcardExample"
                                components={{
                                  model: (
                                    <code className="bg-info/15 px-1 py-0.5 rounded-sm text-xs">
                                      {selectedModel.model_group}
                                    </code>
                                  ),
                                  example: (
                                    <code className="bg-info/15 px-1 py-0.5 rounded-sm text-xs">
                                      {selectedModel.model_group.replaceAll("*", "my-custom-value")}
                                    </code>
                                  ),
                                }}
                              />
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Token and Cost Information */}
                  <div>
                    <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.modelDialog.tokenCost")}</p>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.modelDialog.maxInputTokens")}</p>
                        <p>
                          {selectedModel.max_input_tokens?.toLocaleString() ||
                            t("aiHub.publicModelHub.modelDialog.notSpecified")}
                        </p>
                      </div>
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.modelDialog.maxOutputTokens")}</p>
                        <p>
                          {selectedModel.max_output_tokens?.toLocaleString() ||
                            t("aiHub.publicModelHub.modelDialog.notSpecified")}
                        </p>
                      </div>
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.modelDialog.inputCost")}</p>
                        <p>
                          {selectedModel.input_cost_per_token
                            ? formatCost(selectedModel.input_cost_per_token)
                            : t("aiHub.publicModelHub.modelDialog.notSpecified")}
                        </p>
                      </div>
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.modelDialog.outputCost")}</p>
                        <p>
                          {selectedModel.output_cost_per_token
                            ? formatCost(selectedModel.output_cost_per_token)
                            : t("aiHub.publicModelHub.modelDialog.notSpecified")}
                        </p>
                      </div>
                    </div>
                    <DeploymentPricingBreakdown
                      className="mt-4"
                      deployments={selectedModel.deployment_pricing}
                      heading={
                        <p className="font-medium">{t("aiHub.publicModelHub.modelDialog.pricingByDeployment")}</p>
                      }
                    />
                  </div>

                  {/* Capabilities */}
                  <div>
                    <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.modelDialog.capabilities")}</p>
                    <div className="flex flex-wrap gap-2">
                      {(() => {
                        const capabilities = getModelCapabilities(selectedModel);

                        if (capabilities.length === 0) {
                          return (
                            <p className="text-muted-foreground">
                              {t("aiHub.publicModelHub.modelDialog.noCapabilities")}
                            </p>
                          );
                        }

                        return capabilities.map((capability) => (
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
                      <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.modelDialog.rateLimits")}</p>
                      <div className="grid grid-cols-2 gap-4">
                        {selectedModel.tpm && (
                          <div>
                            <p className="font-medium">{t("aiHub.publicModelHub.modelDialog.tpm")}</p>
                            <p>{selectedModel.tpm.toLocaleString()}</p>
                          </div>
                        )}
                        {selectedModel.rpm && (
                          <div>
                            <p className="font-medium">{t("aiHub.publicModelHub.modelDialog.rpm")}</p>
                            <p>{selectedModel.rpm.toLocaleString()}</p>
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {/* Supported OpenAI Parameters */}
                  {selectedModel.supported_openai_params && selectedModel.supported_openai_params.length > 0 && (
                    <div>
                      <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.modelDialog.openaiParams")}</p>
                      <div className="flex flex-wrap gap-2">
                        {selectedModel.supported_openai_params.map((param) => (
                          <Badge key={param} variant="secondary">
                            {param}
                          </Badge>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Usage Example */}
                  <div>
                    <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.modelDialog.usageExample")}</p>
                    <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                      <pre className="text-sm">{usageExample(selectedModel)}</pre>
                    </div>
                    <div className="mt-2 text-right">
                      <button
                        onClick={() => copyToClipboard(usageExample(selectedModel))}
                        className="text-sm text-info hover:text-info/80 cursor-pointer"
                      >
                        {t("aiHub.publicModelHub.modelDialog.copyToClipboard")}
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </DialogContent>
          </Dialog>

          {/* Agent Details Modal */}
          <Dialog open={isAgentModalVisible} onOpenChange={(open) => !open && handleAgentModalCancel()}>
            <DialogContent
              className={`${isChat ? "chat-theme chat-model-hub-dialog " : ""}max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]`}
            >
              <DialogHeader>
                <DialogTitle className="flex min-w-0 items-center space-x-2">
                  <span className="break-words">
                    {selectedAgent?.name || t("aiHub.publicModelHub.agentDialog.fallbackTitle")}
                  </span>
                  {selectedAgent && (
                    <Tooltip>
                      <TooltipTrigger
                        render={
                          <Copy
                            onClick={() => copyToClipboard(selectedAgent.name)}
                            className="cursor-pointer text-muted-foreground hover:text-info w-4 h-4 shrink-0"
                          />
                        }
                      />
                      <TooltipContent className={portalClassName}>
                        {t("aiHub.publicModelHub.agentDialog.copyName")}
                      </TooltipContent>
                    </Tooltip>
                  )}
                </DialogTitle>
              </DialogHeader>
              {selectedAgent && (
                <div className="space-y-6">
                  {/* Agent Overview */}
                  <div>
                    <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.agentDialog.overview")}</p>
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.agentDialog.name")}</p>
                        <p>{selectedAgent.name}</p>
                      </div>
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.agentDialog.version")}</p>
                        <p>{selectedAgent.version}</p>
                      </div>
                      <div className="col-span-2">
                        <p className="font-medium">{t("aiHub.publicModelHub.agentDialog.description")}</p>
                        <p>{selectedAgent.description}</p>
                      </div>
                      {selectedAgent.url && (
                        <div>
                          <p className="font-medium">{t("aiHub.publicModelHub.agentDialog.url")}</p>
                          <a
                            href={selectedAgent.url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-info hover:text-info/80 text-sm break-all"
                          >
                            {selectedAgent.url}
                          </a>
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Capabilities */}
                  {selectedAgent.capabilities && (
                    <div>
                      <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.agentDialog.capabilities")}</p>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(selectedAgent.capabilities)
                          .filter(([_, value]) => value === true)
                          .map(([key]) => (
                            <Badge key={key} variant="secondary" className="capitalize">
                              {key}
                            </Badge>
                          ))}
                      </div>
                    </div>
                  )}

                  {/* Skills */}
                  {selectedAgent.skills && selectedAgent.skills.length > 0 && (
                    <div>
                      <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.agentDialog.skills")}</p>
                      <div className="space-y-4">
                        {selectedAgent.skills.map((skill, index) => (
                          <div key={index} className="border border-border rounded-lg p-4">
                            <div className="flex items-start justify-between mb-2">
                              <div>
                                <p className="font-medium text-base">{skill.name}</p>
                                <p className="text-sm text-muted-foreground">{skill.description}</p>
                              </div>
                            </div>
                            {skill.tags && skill.tags.length > 0 && (
                              <div className="flex flex-wrap gap-1 mt-2">
                                {skill.tags.map((tag) => (
                                  <Badge key={tag} variant="secondary" className="text-xs">
                                    {tag}
                                  </Badge>
                                ))}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Input/Output Modes */}
                  <div>
                    <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.agentDialog.ioModes")}</p>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.agentDialog.inputModes")}</p>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {(selectedAgent.defaultInputModes ?? []).map((mode) => (
                            <Badge key={mode} variant="secondary">
                              {mode}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.agentDialog.outputModes")}</p>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {(selectedAgent.defaultOutputModes ?? []).map((mode) => (
                            <Badge key={mode} variant="secondary">
                              {mode}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Documentation */}
                  {selectedAgent.documentationUrl && (
                    <div>
                      <p className="text-lg font-semibold mb-4">
                        {t("aiHub.publicModelHub.agentDialog.documentation")}
                      </p>
                      <a
                        href={selectedAgent.documentationUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-info hover:text-info/80 flex items-center space-x-2"
                      >
                        <ExternalLinkIcon className="w-4 h-4" />
                        <span>{t("aiHub.publicModelHub.agentDialog.viewDocumentation")}</span>
                      </a>
                    </div>
                  )}

                  {/* A2A Usage Example */}
                  <div>
                    <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.agentDialog.usageExample")}</p>

                    {/* Step 1: Retrieve Agent Card */}
                    <div className="mb-4">
                      <p className="text-sm font-medium mb-2 text-foreground">
                        {t("aiHub.publicModelHub.agentDialog.stepRetrieveCard")}
                      </p>
                      <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                        <pre className="text-xs">
                          {`base_url = '${selectedAgent.url}'

resolver = A2ACardResolver(
    httpx_client=httpx_client,
    base_url=base_url,
    # agent_card_path uses default, extended_agent_card_path also uses default
)

# Fetch Public Agent Card and Initialize Client
final_agent_card_to_use: AgentCard | None = None
_public_card = (
    await resolver.get_agent_card()
)  # Fetches from default public path - \`/agents/{agent_id}/\`
final_agent_card_to_use = _public_card

if _public_card.supports_authenticated_extended_card:
    try:
        auth_headers_dict = {
            'Authorization': 'Bearer dummy-token-for-extended-card'
        }
        _extended_card = await resolver.get_agent_card(
            relative_card_path=EXTENDED_AGENT_CARD_PATH,
            http_kwargs={'headers': auth_headers_dict},
        )
        final_agent_card_to_use = (
            _extended_card  # Update to use the extended card
        )
    except Exception as e_extended:
        logger.warning(
            f'Failed to fetch extended agent card: {e_extended}. Will proceed with public card.',
            exc_info=True,
        )`}
                        </pre>
                      </div>
                      <div className="mt-2 text-right">
                        <button
                          onClick={() => {
                            const codeSnippet = `from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
    SendStreamingMessageRequest,
)
from a2a.utils.constants import (
    AGENT_CARD_WELL_KNOWN_PATH,
    EXTENDED_AGENT_CARD_PATH,
)

base_url = '${selectedAgent.url}'

resolver = A2ACardResolver(
    httpx_client=httpx_client,
    base_url=base_url,
    # agent_card_path uses default, extended_agent_card_path also uses default
)

# Fetch Public Agent Card and Initialize Client
final_agent_card_to_use: AgentCard | None = None
_public_card = (
    await resolver.get_agent_card()
)  # Fetches from default public path - \`/agents/{agent_id}/\`
final_agent_card_to_use = _public_card

if _public_card.supports_authenticated_extended_card:
    try:
        auth_headers_dict = {
            'Authorization': 'Bearer dummy-token-for-extended-card'
        }
        _extended_card = await resolver.get_agent_card(
            relative_card_path=EXTENDED_AGENT_CARD_PATH,
            http_kwargs={'headers': auth_headers_dict},
        )
        final_agent_card_to_use = (
            _extended_card  # Update to use the extended card
        )
    except Exception as e_extended:
        logger.warning(
            f'Failed to fetch extended agent card: {e_extended}. Will proceed with public card.',
            exc_info=True,
        )`;
                            copyToClipboard(codeSnippet);
                          }}
                          className="text-sm text-info hover:text-info/80 cursor-pointer"
                        >
                          {t("aiHub.publicModelHub.agentDialog.copyToClipboard")}
                        </button>
                      </div>
                    </div>

                    {/* Step 2: Call the Agent */}
                    <div>
                      <p className="text-sm font-medium mb-2 text-foreground">
                        {t("aiHub.publicModelHub.agentDialog.stepCallAgent")}
                      </p>
                      <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                        <pre className="text-xs">
                          {`client = A2AClient(
    httpx_client=httpx_client, agent_card=final_agent_card_to_use
)

send_message_payload: dict[str, Any] = {
    'message': {
        'role': 'user',
        'parts': [
            {'kind': 'text', 'text': 'how much is 10 USD in INR?'}
        ],
        'messageId': uuid4().hex,
    },
}
request = SendMessageRequest(
    id=str(uuid4()), params=MessageSendParams(**send_message_payload)
)

response = await client.send_message(request)
print(response.model_dump(mode='json', exclude_none=True))`}
                        </pre>
                      </div>
                      <div className="mt-2 text-right">
                        <button
                          onClick={() => {
                            const codeSnippet = `client = A2AClient(
    httpx_client=httpx_client, agent_card=final_agent_card_to_use
)

send_message_payload: dict[str, Any] = {
    'message': {
        'role': 'user',
        'parts': [
            {'kind': 'text', 'text': 'how much is 10 USD in INR?'}
        ],
        'messageId': uuid4().hex,
    },
}
request = SendMessageRequest(
    id=str(uuid4()), params=MessageSendParams(**send_message_payload)
)

response = await client.send_message(request)
print(response.model_dump(mode='json', exclude_none=True))`;
                            copyToClipboard(codeSnippet);
                          }}
                          className="text-sm text-info hover:text-info/80 cursor-pointer"
                        >
                          {t("aiHub.publicModelHub.agentDialog.copyToClipboard")}
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </DialogContent>
          </Dialog>

          {/* MCP Server Details Modal */}
          <Dialog open={isMcpModalVisible} onOpenChange={(open) => !open && handleMcpModalCancel()}>
            <DialogContent
              className={`${isChat ? "chat-theme chat-model-hub-dialog " : ""}max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]`}
            >
              <DialogHeader>
                <DialogTitle className="flex min-w-0 items-center space-x-2">
                  <span className="break-words">
                    {selectedMcpServer?.server_name || t("aiHub.publicModelHub.mcpDialog.fallbackTitle")}
                  </span>
                  {selectedMcpServer && (
                    <Tooltip>
                      <TooltipTrigger
                        render={
                          <Copy
                            onClick={() => copyToClipboard(selectedMcpServer.server_name)}
                            className="cursor-pointer text-muted-foreground hover:text-info w-4 h-4 shrink-0"
                          />
                        }
                      />
                      <TooltipContent className={portalClassName}>
                        {t("aiHub.publicModelHub.mcpDialog.copyName")}
                      </TooltipContent>
                    </Tooltip>
                  )}
                </DialogTitle>
              </DialogHeader>
              {selectedMcpServer && (
                <div className="space-y-6">
                  {/* Server Overview */}
                  <div>
                    <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.mcpDialog.overview")}</p>
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.mcpDialog.serverName")}</p>
                        <p>{selectedMcpServer.server_name}</p>
                      </div>
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.mcpDialog.transport")}</p>
                        <Badge variant="secondary">{selectedMcpServer.transport}</Badge>
                      </div>
                      {selectedMcpServer.alias && (
                        <div>
                          <p className="font-medium">{t("aiHub.publicModelHub.mcpDialog.alias")}</p>
                          <p>{selectedMcpServer.alias}</p>
                        </div>
                      )}
                      <div>
                        <p className="font-medium">{t("aiHub.publicModelHub.mcpDialog.authType")}</p>
                        <Badge variant={selectedMcpServer.auth_type === "none" ? "outline" : "secondary"}>
                          {selectedMcpServer.auth_type}
                        </Badge>
                      </div>
                      <div className="col-span-2">
                        <p className="font-medium">{t("aiHub.publicModelHub.mcpDialog.description")}</p>
                        <p>{selectedMcpServer.mcp_info?.description || "-"}</p>
                      </div>
                    </div>
                  </div>

                  {/* Additional Info */}
                  {selectedMcpServer.mcp_info && Object.keys(selectedMcpServer.mcp_info).length > 0 && (
                    <div>
                      <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.mcpDialog.additionalInfo")}</p>
                      <div className="bg-muted p-4 rounded-lg">
                        <pre className="text-xs overflow-x-auto">
                          {JSON.stringify(selectedMcpServer.mcp_info, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* Usage Example */}
                  <div>
                    <p className="text-lg font-semibold mb-4">{t("aiHub.publicModelHub.mcpDialog.usageExample")}</p>
                    <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                      <pre className="text-sm">
                        {`# Using MCP Server with Python FastMCP

from fastmcp import Client
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
                      </pre>
                    </div>
                    <div className="mt-2 text-right">
                      <button
                        onClick={() => {
                          const codeSnippet = `# Using MCP Server with Python FastMCP

from fastmcp import Client
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
    asyncio.run(main())`;
                          copyToClipboard(codeSnippet);
                        }}
                        className="text-sm text-info hover:text-info/80 cursor-pointer"
                      >
                        {t("aiHub.publicModelHub.mcpDialog.copyToClipboard")}
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </DialogContent>
          </Dialog>
        </div>
      </TooltipProvider>
    </ThemeProvider>
  );
};

export default PublicModelHub;
