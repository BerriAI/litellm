import { ThemeProvider } from "@/contexts/ThemeContext";
import { ExternalLinkIcon, SearchIcon } from "@heroicons/react/outline";
import { SortingState } from "@tanstack/react-table";
import { Copy, Inbox, Info } from "lucide-react";
import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
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
import { DataTable } from "./shared/DataTable";
import { toast } from "@/lib/toast";
import Navbar from "./navbar";
import {
  agentHubPublicModelsCall,
  skillHubPublicCall,
  getProxyBaseUrl,
  getPublicModelHubInfo,
  getUiConfig,
  mcpHubPublicServersCall,
  modelHubPublicModelsCall,
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

interface PublicModelHubProps {
  accessToken?: string | null;
  isEmbedded?: boolean; // When true, hides navbar and adjusts layout for embedding in dashboard
}

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

const PublicModelHub: React.FC<PublicModelHubProps> = ({ accessToken, isEmbedded = false }) => {
  const anchor = useComboboxAnchor();
  const [modelHubData, setModelHubData] = useState<ModelGroupInfo[] | null>(null);
  const [agentHubData, setAgentHubData] = useState<AgentCard[] | null>(null);
  const [mcpHubData, setMcpHubData] = useState<MCPServerData[] | null>(null);
  const [pageTitle, setPageTitle] = useState<string>("LiteLLM Gateway");
  const [customDocsDescription, setCustomDocsDescription] = useState<string | null>(null);
  const [litellmVersion, setLitellmVersion] = useState<string>("");
  const [usefulLinks, setUsefulLinks] = useState<Record<string, string | { url: string; index: number }>>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [agentLoading, setAgentLoading] = useState<boolean>(true);
  const [mcpLoading, setMcpLoading] = useState<boolean>(true);
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [agentSearchTerm, setAgentSearchTerm] = useState<string>("");
  const [mcpSearchTerm, setMcpSearchTerm] = useState<string>("");
  const [selectedProviders, setSelectedProviders] = useState<string[]>([]);
  const [selectedModes, setSelectedModes] = useState<string[]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [selectedAgentSkills, setSelectedAgentSkills] = useState<string[]>([]);
  const [selectedMcpTransports, setSelectedMcpTransports] = useState<string[]>([]);
  const [serviceStatus, setServiceStatus] = useState<string>("I'm alive! ✓");
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

      const fetchPublicData = async () => {
        try {
          setLoading(true);
          const _modelHubData = await modelHubPublicModelsCall();
          setModelHubData(Array.isArray(_modelHubData) ? _modelHubData : []);
        } catch (error) {
          console.error("There was an error fetching the public model data", error);
          setServiceStatus("Service unavailable");
        } finally {
          setLoading(false);
        }
      };

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

      fetchPublicData();
      fetchAgentData();
      fetchMcpData();
      fetchSkillData();
    };

    initializeAndFetch();
  }, []);

  // Clear filters when filter values change to avoid confusion
  useEffect(() => {
    // This would clear selections if we had any selection functionality
    // For now, it's just for consistency with the original component
  }, [searchTerm, selectedProviders, selectedModes, selectedFeatures]);

  const getUniqueProviders = (data: ModelGroupInfo[]) => {
    const providers = new Set<string>();
    data.forEach((model) => {
      (model.providers ?? []).forEach((provider) => providers.add(provider));
    });
    return Array.from(providers);
  };

  const getUniqueModes = (data: ModelGroupInfo[]) => {
    const modes = new Set<string>();
    data.forEach((model) => {
      if (model.mode) modes.add(model.mode);
    });
    return Array.from(modes);
  };

  const getUniqueFeatures = (data: ModelGroupInfo[]) => {
    const features = new Set<string>();
    data.forEach((model) => {
      // Find all properties that start with 'supports_' and are true
      Object.entries(model)
        .filter(([key, value]) => key.startsWith("supports_") && value === true)
        .forEach(([key]) => {
          // Format the feature name (remove 'supports_' prefix and convert to title case)
          const featureName = key
            .replace(/^supports_/, "")
            .split("_")
            .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
            .join(" ");
          features.add(featureName);
        });
    });
    return Array.from(features).sort();
  };

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

  const filteredData = useMemo(() => {
    if (!modelHubData || !Array.isArray(modelHubData)) return [];

    let searchResults = modelHubData;

    // Apply search if there's a search term
    if (searchTerm.trim()) {
      const lowercaseSearch = searchTerm.toLowerCase();
      const searchWords = lowercaseSearch.split(/\s+/);

      // First, try flexible matching that handles different separators
      const exactMatches = modelHubData.filter((model) => {
        const modelName = model.model_group.toLowerCase();

        // Check if it contains the exact search term
        if (modelName.includes(lowercaseSearch)) {
          return true;
        }

        // Check if it contains all search words (handles spaces vs slashes/dashes)
        return searchWords.every((word) => modelName.includes(word));
      });

      // If we have exact matches, rank them by relevance
      if (exactMatches.length > 0) {
        searchResults = exactMatches.sort((a, b) => {
          const aName = a.model_group.toLowerCase();
          const bName = b.model_group.toLowerCase();

          // Calculate relevance scores
          const aExactMatch = aName === lowercaseSearch ? 1000 : 0;
          const bExactMatch = bName === lowercaseSearch ? 1000 : 0;

          const aStartsWith = aName.startsWith(lowercaseSearch) ? 100 : 0;
          const bStartsWith = bName.startsWith(lowercaseSearch) ? 100 : 0;

          const aContainsWords = lowercaseSearch.split(/\s+/).every((word) => aName.includes(word)) ? 50 : 0;
          const bContainsWords = lowercaseSearch.split(/\s+/).every((word) => bName.includes(word)) ? 50 : 0;

          const aLength = aName.length;
          const bLength = bName.length;

          const aScore = aExactMatch + aStartsWith + aContainsWords + (1000 - aLength);
          const bScore = bExactMatch + bStartsWith + bContainsWords + (1000 - bLength);

          return bScore - aScore; // Higher score first
        });
      }
    }

    // Apply other filters
    return searchResults.filter((model) => {
      const matchesProvider =
        selectedProviders.length === 0 || selectedProviders.some((provider) => model.providers.includes(provider));
      const matchesMode = selectedModes.length === 0 || selectedModes.includes(model.mode || "");

      // Check if model has any of the selected features
      const matchesFeature =
        selectedFeatures.length === 0 ||
        Object.entries(model)
          .filter(([key, value]) => key.startsWith("supports_") && value === true)
          .some(([key]) => {
            const featureName = key
              .replace(/^supports_/, "")
              .split("_")
              .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
              .join(" ");
            return selectedFeatures.includes(featureName);
          });

      return matchesProvider && matchesMode && matchesFeature;
    });
  }, [modelHubData, searchTerm, selectedProviders, selectedModes, selectedFeatures]);

  const filteredAgentData = useMemo(() => {
    if (!agentHubData || !Array.isArray(agentHubData)) return [];

    let searchResults = agentHubData;

    // Apply search if there's a search term
    if (agentSearchTerm.trim()) {
      const lowercaseSearch = agentSearchTerm.toLowerCase();
      const searchWords = lowercaseSearch.split(/\s+/);

      searchResults = agentHubData.filter((agent) => {
        const agentName = agent.name.toLowerCase();
        const agentDescription = agent.description.toLowerCase();

        // Check if it contains the exact search term
        if (agentName.includes(lowercaseSearch) || agentDescription.includes(lowercaseSearch)) {
          return true;
        }

        // Check if it contains all search words
        return searchWords.every((word) => agentName.includes(word) || agentDescription.includes(word));
      });

      // Sort by relevance
      searchResults = searchResults.sort((a, b) => {
        const aName = a.name.toLowerCase();
        const bName = b.name.toLowerCase();

        const aExactMatch = aName === lowercaseSearch ? 1000 : 0;
        const bExactMatch = bName === lowercaseSearch ? 1000 : 0;

        const aStartsWith = aName.startsWith(lowercaseSearch) ? 100 : 0;
        const bStartsWith = bName.startsWith(lowercaseSearch) ? 100 : 0;

        const aScore = aExactMatch + aStartsWith + (1000 - aName.length);
        const bScore = bExactMatch + bStartsWith + (1000 - bName.length);

        return bScore - aScore;
      });
    }

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

    let searchResults = mcpHubData;

    // Apply search if there's a search term
    if (mcpSearchTerm.trim()) {
      const lowercaseSearch = mcpSearchTerm.toLowerCase();
      const searchWords = lowercaseSearch.split(/\s+/);

      searchResults = mcpHubData.filter((server) => {
        const serverName = server.server_name.toLowerCase();
        const serverDescription = (server.mcp_info?.description || "").toLowerCase();

        // Check if it contains the exact search term
        if (serverName.includes(lowercaseSearch) || serverDescription.includes(lowercaseSearch)) {
          return true;
        }

        // Check if it contains all search words
        return searchWords.every((word) => serverName.includes(word) || serverDescription.includes(word));
      });

      // Sort by relevance
      searchResults = searchResults.sort((a, b) => {
        const aName = a.server_name.toLowerCase();
        const bName = b.server_name.toLowerCase();

        const aExactMatch = aName === lowercaseSearch ? 1000 : 0;
        const bExactMatch = bName === lowercaseSearch ? 1000 : 0;

        const aStartsWith = aName.startsWith(lowercaseSearch) ? 100 : 0;
        const bStartsWith = bName.startsWith(lowercaseSearch) ? 100 : 0;

        const aScore = aExactMatch + aStartsWith + (1000 - aName.length);
        const bScore = bExactMatch + bStartsWith + (1000 - bName.length);

        return bScore - aScore;
      });
    }

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
    toast.success("Copied to clipboard!");
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

  const [modelSorting, setModelSorting] = useState<SortingState>([{ id: "model_group", desc: false }]);
  const [agentSorting, setAgentSorting] = useState<SortingState>([{ id: "name", desc: false }]);
  const [mcpSorting, setMcpSorting] = useState<SortingState>([{ id: "server_name", desc: false }]);

  const modelColumns = useMemo(() => getPublicModelHubColumns({ onModelClick: showModal }), [showModal]);
  const agentColumns = useMemo(() => getPublicAgentHubColumns({ onAgentClick: showAgentModal }), [showAgentModal]);
  const mcpColumns = useMemo(() => getPublicMCPHubColumns({ onServerClick: showMcpModal }), [showMcpModal]);

  const hasAgents = Array.isArray(agentHubData) && agentHubData.length > 0;
  const hasMcpServers = Array.isArray(mcpHubData) && mcpHubData.length > 0;

  const providerOptions = useMemo(
    () => (Array.isArray(modelHubData) ? getUniqueProviders(modelHubData) : []),
    [modelHubData],
  );
  const modeOptions = useMemo(
    () =>
      Array.isArray(modelHubData) ? getUniqueModes(modelHubData).map((mode) => ({ label: mode, value: mode })) : [],
    [modelHubData],
  );
  const featureOptions = useMemo(
    () =>
      Array.isArray(modelHubData)
        ? getUniqueFeatures(modelHubData).map((feature) => ({ label: feature, value: feature }))
        : [],
    [modelHubData],
  );
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
        <div className={isEmbedded ? "w-full" : "min-h-screen bg-card"}>
          {/* Navigation - only show when not embedded */}
          {!isEmbedded && <Navbar accessToken={accessToken || null} isPublicPage={true} />}

          <div className={isEmbedded ? "w-full p-6" : "w-full px-8 py-12"}>
            {/* Embedded Explainer - only shown when embedded in dashboard */}
            {isEmbedded && (
              <div className="mb-6 p-4 bg-info/10 border border-info/20 rounded-lg">
                <p className="text-sm text-foreground">
                  These are models, agents, and MCP servers your proxy admin has indicated are available in your
                  company.
                </p>
              </div>
            )}

            {/* About Section - only shown when not embedded */}
            {!isEmbedded && (
              <Card className="mb-10 p-8 bg-card border border-border rounded-lg shadow-xs">
                <h2 className="text-2xl font-semibold mb-6 text-foreground">About</h2>
                <p className="text-foreground mb-6 text-base leading-relaxed">
                  {customDocsDescription
                    ? customDocsDescription
                    : "Proxy Server to call 100+ LLMs in the OpenAI format."}
                </p>
                <div className="flex items-center space-x-3 text-sm text-muted-foreground">
                  <span className="flex items-center">
                    <span className="w-4 h-4 mr-2">🔧</span>
                    Built with litellm: v{litellmVersion}
                  </span>
                </div>
              </Card>
            )}

            {/* Useful Links - only shown when not embedded */}
            {usefulLinks && Object.keys(usefulLinks).length > 0 && (
              <Card className="mb-10 p-8 bg-card border border-border rounded-lg shadow-xs">
                <h2 className="text-2xl font-semibold mb-6 text-foreground">Useful Links</h2>
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
            {!isEmbedded && (
              <Card className="mb-10 p-8 bg-card border border-border rounded-lg shadow-xs">
                <h2 className="text-2xl font-semibold mb-6 text-foreground">Health and Endpoint Status</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <p className="text-success font-medium text-sm">Service status: {serviceStatus}</p>
                </div>
              </Card>
            )}

            {/* Tabs for Models and Agents */}
            <Card className="p-8 bg-card border border-border rounded-lg shadow-xs">
              <Tabs value={activeTab} onValueChange={setActiveTab} className="public-hub-tabs">
                <TabsList>
                  <TabsTrigger value="models">Model Hub</TabsTrigger>
                  {hasAgents && <TabsTrigger value="agents">Agent Hub</TabsTrigger>}
                  {hasMcpServers && <TabsTrigger value="mcp">MCP Hub</TabsTrigger>}
                  <TabsTrigger value="skills">Skill Hub</TabsTrigger>
                </TabsList>

                {/* Models Tab */}
                <TabsContent value="models">
                  <div className="flex justify-between items-center mb-8">
                    <h2 className="text-2xl font-semibold text-foreground">Available Models</h2>
                  </div>

                  {/* Filters */}
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8 p-6 bg-muted rounded-lg border border-border">
                    <div>
                      <div className="flex items-center space-x-2 mb-3">
                        <p className="text-sm font-medium text-foreground">Search Models:</p>
                        <Tooltip>
                          <TooltipTrigger render={<Info className="w-4 h-4 text-muted-foreground cursor-help" />} />
                          <TooltipContent side="top">
                            Smart search with relevance ranking - finds models containing your search terms, ranked by
                            relevance. Try searching &apos;xai grok-4&apos;, &apos;claude-4&apos;, &apos;gpt-4&apos;, or
                            &apos;sonnet&apos;
                          </TooltipContent>
                        </Tooltip>
                      </div>
                      <div className="relative">
                        <SearchIcon className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 transform -translate-y-1/2" />
                        <input
                          type="text"
                          placeholder="Search model names... (smart search enabled)"
                          value={searchTerm}
                          onChange={(e) => setSearchTerm(e.target.value)}
                          className="border border-border rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-hidden focus:ring-2 focus:ring-ring focus:border-transparent bg-card"
                        />
                      </div>
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-medium mb-3 text-foreground">Provider:</p>
                      <Combobox
                        multiple
                        items={providerOptions}
                        value={selectedProviders}
                        onValueChange={(values: string[]) => setSelectedProviders(values)}
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
                            placeholder="Select providers"
                            aria-label="Select providers"
                            className="min-w-24"
                          />
                        </ComboboxChips>
                        <ComboboxContent anchor={anchor}>
                          <ComboboxEmpty>No providers found</ComboboxEmpty>
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
                    <div className="min-w-0">
                      <p className="text-sm font-medium mb-3 text-foreground">Mode:</p>
                      <MultiSelect
                        options={modeOptions}
                        value={selectedModes}
                        onValueChange={setSelectedModes}
                        placeholder="Select modes"
                        className="w-full"
                      />
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-medium mb-3 text-foreground">Features:</p>
                      <MultiSelect
                        options={featureOptions}
                        value={selectedFeatures}
                        onValueChange={setSelectedFeatures}
                        placeholder="Select features"
                        className="w-full"
                      />
                    </div>
                  </div>

                  <DataTable
                    data={filteredData}
                    columns={modelColumns}
                    getRowId={(model, index) => model.model_group || String(index)}
                    sortingMode="client"
                    sorting={modelSorting}
                    onSortingChange={setModelSorting}
                    isLoading={loading}
                    loadingMessage="Loading models…"
                    noDataMessage={
                      <PublicHubEmptyState
                        title={modelHubData?.length ? "No matching models" : "No models available"}
                        body={
                          modelHubData?.length
                            ? "Adjust the search or filters to see more models."
                            : "Models made public by the proxy admin will appear here."
                        }
                      />
                    }
                    size="compact"
                  />

                  <div className="mt-8 text-center">
                    <p className="text-sm text-muted-foreground">
                      Showing {filteredData.length} of {modelHubData?.length || 0} models
                    </p>
                  </div>
                </TabsContent>

                {/* Agents Tab */}
                {hasAgents && (
                  <TabsContent value="agents">
                    <div className="flex justify-between items-center mb-8">
                      <h2 className="text-2xl font-semibold text-foreground">Available Agents</h2>
                    </div>

                    {/* Filters */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-muted rounded-lg border border-border">
                      <div>
                        <div className="flex items-center space-x-2 mb-3">
                          <p className="text-sm font-medium text-foreground">Search Agents:</p>
                          <Tooltip>
                            <TooltipTrigger render={<Info className="w-4 h-4 text-muted-foreground cursor-help" />} />
                            <TooltipContent side="top">Search agents by name or description</TooltipContent>
                          </Tooltip>
                        </div>
                        <div className="relative">
                          <SearchIcon className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 transform -translate-y-1/2" />
                          <input
                            type="text"
                            placeholder="Search agent names or descriptions..."
                            value={agentSearchTerm}
                            onChange={(e) => setAgentSearchTerm(e.target.value)}
                            className="border border-border rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-hidden focus:ring-2 focus:ring-ring focus:border-transparent bg-card"
                          />
                        </div>
                      </div>
                      <div className="min-w-0">
                        <p className="text-sm font-medium mb-3 text-foreground">Skills:</p>
                        <MultiSelect
                          options={agentSkillOptions}
                          value={selectedAgentSkills}
                          onValueChange={setSelectedAgentSkills}
                          placeholder="Select skills"
                          className="w-full"
                        />
                      </div>
                    </div>

                    <DataTable
                      data={filteredAgentData}
                      columns={agentColumns}
                      getRowId={(agent, index) => agent.name || String(index)}
                      sortingMode="client"
                      sorting={agentSorting}
                      onSortingChange={setAgentSorting}
                      isLoading={agentLoading}
                      loadingMessage="Loading agents…"
                      noDataMessage={
                        <PublicHubEmptyState
                          title="No matching agents"
                          body="Adjust the search or skill filter to see more agents."
                        />
                      }
                      size="compact"
                    />

                    <div className="mt-8 text-center">
                      <p className="text-sm text-muted-foreground">
                        Showing {filteredAgentData.length} of {agentHubData?.length || 0} agents
                      </p>
                    </div>
                  </TabsContent>
                )}

                {/* MCP Servers Tab */}
                {hasMcpServers && (
                  <TabsContent value="mcp">
                    <div className="flex justify-between items-center mb-8">
                      <h2 className="text-2xl font-semibold text-foreground">Available MCP Servers</h2>
                    </div>

                    {/* Filters */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8 p-6 bg-muted rounded-lg border border-border">
                      <div>
                        <div className="flex items-center space-x-2 mb-3">
                          <p className="text-sm font-medium text-foreground">Search MCP Servers:</p>
                          <Tooltip>
                            <TooltipTrigger render={<Info className="w-4 h-4 text-muted-foreground cursor-help" />} />
                            <TooltipContent side="top">Search MCP servers by name or description</TooltipContent>
                          </Tooltip>
                        </div>
                        <div className="relative">
                          <SearchIcon className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 transform -translate-y-1/2" />
                          <input
                            type="text"
                            placeholder="Search MCP server names or descriptions..."
                            value={mcpSearchTerm}
                            onChange={(e) => setMcpSearchTerm(e.target.value)}
                            className="border border-border rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-hidden focus:ring-2 focus:ring-ring focus:border-transparent bg-card"
                          />
                        </div>
                      </div>
                      <div className="min-w-0">
                        <p className="text-sm font-medium mb-3 text-foreground">Transport:</p>
                        <MultiSelect
                          options={mcpTransportOptions}
                          value={selectedMcpTransports}
                          onValueChange={setSelectedMcpTransports}
                          placeholder="Select transport types"
                          className="w-full"
                        />
                      </div>
                    </div>

                    <DataTable
                      data={filteredMcpData}
                      columns={mcpColumns}
                      getRowId={(server, index) => server.server_id || String(index)}
                      sortingMode="client"
                      sorting={mcpSorting}
                      onSortingChange={setMcpSorting}
                      isLoading={mcpLoading}
                      loadingMessage="Loading MCP servers…"
                      noDataMessage={
                        <PublicHubEmptyState
                          title="No matching MCP servers"
                          body="Adjust the search or transport filter to see more servers."
                        />
                      }
                      size="compact"
                    />

                    <div className="mt-8 text-center">
                      <p className="text-sm text-muted-foreground">
                        Showing {filteredMcpData.length} of {mcpHubData?.length || 0} MCP servers
                      </p>
                    </div>
                  </TabsContent>
                )}

                {/* Skill Hub Tab */}
                <TabsContent value="skills">
                  <SkillHubDashboard skills={skillHubData} isLoading={skillLoading} publicPage={true} />
                </TabsContent>
              </Tabs>
            </Card>
          </div>

          {/* Model Details Modal */}
          <Dialog open={isModalVisible} onOpenChange={(open) => !open && handleModalCancel()}>
            <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]">
              <DialogHeader>
                <DialogTitle className="flex min-w-0 items-center space-x-2">
                  <span className="break-words">{selectedModel?.model_group || "Model Details"}</span>
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
                      <TooltipContent>Copy model name</TooltipContent>
                    </Tooltip>
                  )}
                </DialogTitle>
              </DialogHeader>
              {selectedModel && (
                <div className="space-y-6">
                  {/* Model Overview */}
                  <div>
                    <p className="text-lg font-semibold mb-4">Model Overview</p>
                    <div className="grid grid-cols-2 gap-4 mb-4">
                      <div>
                        <p className="font-medium">Model Name:</p>
                        <p>{selectedModel.model_group}</p>
                      </div>
                      <div>
                        <p className="font-medium">Mode:</p>
                        <p>{selectedModel.mode || "Not specified"}</p>
                      </div>
                      <div>
                        <p className="font-medium">Providers:</p>
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
                    </div>

                    {/* Wildcard Routing Note */}
                    {selectedModel.model_group.includes("*") && (
                      <div className="bg-info/10 border border-info/20 rounded-lg p-4 mb-4">
                        <div className="flex items-start space-x-2">
                          <Info className="w-4 h-4 text-info mt-0.5 shrink-0" />
                          <div>
                            <p className="font-medium text-info mb-2">Wildcard Routing</p>
                            <p className="text-sm text-info mb-2">
                              This model uses wildcard routing. You can pass any value where you see the{" "}
                              <code className="bg-info/15 px-1 py-0.5 rounded-sm text-xs">*</code> symbol.
                            </p>
                            <p className="text-sm text-info">
                              For example, with{" "}
                              <code className="bg-info/15 px-1 py-0.5 rounded-sm text-xs">
                                {selectedModel.model_group}
                              </code>
                              , you can use any string (
                              <code className="bg-info/15 px-1 py-0.5 rounded-sm text-xs">
                                {selectedModel.model_group.replaceAll("*", "my-custom-value")}
                              </code>
                              ) that matches this pattern.
                            </p>
                          </div>
                        </div>
                      </div>
                    )}
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

                        if (capabilities.length === 0) {
                          return <p className="text-muted-foreground">No special capabilities listed</p>;
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
                  {selectedModel.supported_openai_params && selectedModel.supported_openai_params.length > 0 && (
                    <div>
                      <p className="text-lg font-semibold mb-4">Supported OpenAI Parameters</p>
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
                    <p className="text-lg font-semibold mb-4">Usage Example</p>
                    <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                      <pre className="text-sm">
                        {(() => {
                          const codeSnippet = generateCodeSnippet({
                            apiKeySource: "custom",
                            accessToken: null,
                            apiKey: "your_api_key",
                            inputMessage: "Hello, how are you?",
                            chatHistory: [
                              { role: "user", content: "Hello, how are you?", isImage: false } as MessageType,
                            ],
                            selectedTags: [],
                            selectedVectorStores: [],
                            selectedGuardrails: [],
                            selectedPolicies: [],
                            selectedMCPServers: [],
                            endpointType: getEndpointType(selectedModel.mode || "chat"),
                            selectedModel: selectedModel.model_group,
                            selectedSdk: "openai",
                          });
                          return codeSnippet;
                        })()}
                      </pre>
                    </div>
                    <div className="mt-2 text-right">
                      <button
                        onClick={() => {
                          const codeSnippet = generateCodeSnippet({
                            apiKeySource: "custom",
                            accessToken: null,
                            apiKey: "your_api_key",
                            inputMessage: "Hello, how are you?",
                            chatHistory: [
                              { role: "user", content: "Hello, how are you?", isImage: false } as MessageType,
                            ],
                            selectedTags: [],
                            selectedVectorStores: [],
                            selectedGuardrails: [],
                            selectedPolicies: [],
                            selectedMCPServers: [],
                            endpointType: getEndpointType(selectedModel.mode || "chat"),
                            selectedModel: selectedModel.model_group,
                            selectedSdk: "openai",
                          });
                          copyToClipboard(codeSnippet);
                        }}
                        className="text-sm text-info hover:text-info/80 cursor-pointer"
                      >
                        Copy to clipboard
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </DialogContent>
          </Dialog>

          {/* Agent Details Modal */}
          <Dialog open={isAgentModalVisible} onOpenChange={(open) => !open && handleAgentModalCancel()}>
            <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]">
              <DialogHeader>
                <DialogTitle className="flex min-w-0 items-center space-x-2">
                  <span className="break-words">{selectedAgent?.name || "Agent Details"}</span>
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
                      <TooltipContent>Copy agent name</TooltipContent>
                    </Tooltip>
                  )}
                </DialogTitle>
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
                        <p>{selectedAgent.version}</p>
                      </div>
                      <div className="col-span-2">
                        <p className="font-medium">Description:</p>
                        <p>{selectedAgent.description}</p>
                      </div>
                      {selectedAgent.url && (
                        <div>
                          <p className="font-medium">URL:</p>
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
                      <p className="text-lg font-semibold mb-4">Capabilities</p>
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
                      <p className="text-lg font-semibold mb-4">Skills</p>
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
                    <p className="text-lg font-semibold mb-4">Input/Output Modes</p>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="font-medium">Input Modes:</p>
                        <div className="flex flex-wrap gap-1 mt-1">
                          {(selectedAgent.defaultInputModes ?? []).map((mode) => (
                            <Badge key={mode} variant="secondary">
                              {mode}
                            </Badge>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="font-medium">Output Modes:</p>
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
                      <p className="text-lg font-semibold mb-4">Documentation</p>
                      <a
                        href={selectedAgent.documentationUrl}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-info hover:text-info/80 flex items-center space-x-2"
                      >
                        <ExternalLinkIcon className="w-4 h-4" />
                        <span>View Documentation</span>
                      </a>
                    </div>
                  )}

                  {/* A2A Usage Example */}
                  <div>
                    <p className="text-lg font-semibold mb-4">Usage Example (A2A Protocol)</p>

                    {/* Step 1: Retrieve Agent Card */}
                    <div className="mb-4">
                      <p className="text-sm font-medium mb-2 text-foreground">Step 1: Retrieve Agent Card</p>
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
                          Copy to clipboard
                        </button>
                      </div>
                    </div>

                    {/* Step 2: Call the Agent */}
                    <div>
                      <p className="text-sm font-medium mb-2 text-foreground">Step 2: Call the Agent</p>
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
                          Copy to clipboard
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
            <DialogContent className="max-h-[calc(100dvh-2rem)] overflow-y-auto sm:max-w-[1000px]">
              <DialogHeader>
                <DialogTitle className="flex min-w-0 items-center space-x-2">
                  <span className="break-words">{selectedMcpServer?.server_name || "MCP Server Details"}</span>
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
                      <TooltipContent>Copy server name</TooltipContent>
                    </Tooltip>
                  )}
                </DialogTitle>
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
                        <p className="font-medium">Transport:</p>
                        <Badge variant="secondary">{selectedMcpServer.transport}</Badge>
                      </div>
                      {selectedMcpServer.alias && (
                        <div>
                          <p className="font-medium">Alias:</p>
                          <p>{selectedMcpServer.alias}</p>
                        </div>
                      )}
                      <div>
                        <p className="font-medium">Auth Type:</p>
                        <Badge variant={selectedMcpServer.auth_type === "none" ? "outline" : "secondary"}>
                          {selectedMcpServer.auth_type}
                        </Badge>
                      </div>
                      <div className="col-span-2">
                        <p className="font-medium">Description:</p>
                        <p>{selectedMcpServer.mcp_info?.description || "-"}</p>
                      </div>
                    </div>
                  </div>

                  {/* Additional Info */}
                  {selectedMcpServer.mcp_info && Object.keys(selectedMcpServer.mcp_info).length > 0 && (
                    <div>
                      <p className="text-lg font-semibold mb-4">Additional Information</p>
                      <div className="bg-muted p-4 rounded-lg">
                        <pre className="text-xs overflow-x-auto">
                          {JSON.stringify(selectedMcpServer.mcp_info, null, 2)}
                        </pre>
                      </div>
                    </div>
                  )}

                  {/* Usage Example */}
                  <div>
                    <p className="text-lg font-semibold mb-4">Usage Example</p>
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
    asyncio.run(main())`;
                          copyToClipboard(codeSnippet);
                        }}
                        className="text-sm text-info hover:text-info/80 cursor-pointer"
                      >
                        Copy to clipboard
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
