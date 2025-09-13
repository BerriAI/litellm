import React, { useEffect, useState, useRef, useMemo } from "react";
import { modelHubPublicModelsCall, proxyBaseUrl, getUiConfig, getPublicModelHubInfo } from "./networking";
import { ModelDataTable } from "./model_dashboard/table";
import { ColumnDef } from "@tanstack/react-table";
import {
  Card,
  Text,
  Title,
  Button,
} from "@tremor/react";
import { message, Tag, Tooltip, Modal, Select } from "antd";
import { CopyOutlined } from "@ant-design/icons";
import { ExternalLinkIcon, SearchIcon, EyeIcon, CogIcon } from "@heroicons/react/outline";
import { Copy, Info } from "lucide-react";
import { Table as TableInstance } from '@tanstack/react-table';
import { generateCodeSnippet } from "./chat_ui/CodeSnippets";
import { EndpointType, getEndpointType } from "./chat_ui/mode_endpoint_mapping";
import { MessageType } from "./chat_ui/types";
import { getProviderLogoAndName } from "./provider_info_helpers";
import Navbar from "./navbar";
import { ThemeProvider } from "@/contexts/ThemeContext";
import NotificationsManager from "./molecules/notifications_manager";
// Simple approach without react-markdown dependency

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
  [key: string]: any;
}

interface PublicModelHubProps {
  accessToken?: string | null;
}

const PublicModelHub: React.FC<PublicModelHubProps> = ({ accessToken }) => {
  const [modelHubData, setModelHubData] = useState<ModelGroupInfo[] | null>(null);
  const [pageTitle, setPageTitle] = useState<string>("LiteLLM Gateway");
  const [customDocsDescription, setCustomDocsDescription] = useState<string | null>(null);
  const [litellmVersion, setLitellmVersion] = useState<string>("");
  const [usefulLinks, setUsefulLinks] = useState<Record<string, string>>({});
  const [loading, setLoading] = useState<boolean>(true);
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [selectedProviders, setSelectedProviders] = useState<string[]>([]);
  const [selectedModes, setSelectedModes] = useState<string[]>([]);
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([]);
  const [serviceStatus, setServiceStatus] = useState<string>("I'm alive! ✓");
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedModel, setSelectedModel] = useState<null | ModelGroupInfo>(null);
  const [proxySettings, setProxySettings] = useState<any>({});
  const tableRef = useRef<TableInstance<any>>(null);

  useEffect(() => {
    const fetchPublicData = async () => {
      try {
        setLoading(true);
        const _modelHubData = await modelHubPublicModelsCall();
        console.log("ModelHubData:", _modelHubData);
        setModelHubData(_modelHubData);
      } catch (error) {
        console.error("There was an error fetching the public model data", error);
        setServiceStatus("Service unavailable");
      } finally {
        setLoading(false);
      }
    };

    const fetchPublicModelHubInfo = async () => {
      const publicModelHubInfo = await getPublicModelHubInfo();
      console.log("Public Model Hub Info:", publicModelHubInfo);
      setPageTitle(publicModelHubInfo.docs_title);
      setCustomDocsDescription(publicModelHubInfo.custom_docs_description);
      setLitellmVersion(publicModelHubInfo.litellm_version);
      setUsefulLinks(publicModelHubInfo.useful_links || {});
    };


    fetchPublicModelHubInfo();

    fetchPublicData();
  }, []);

  // Clear filters when filter values change to avoid confusion
  useEffect(() => {
    // This would clear selections if we had any selection functionality
    // For now, it's just for consistency with the original component
  }, [searchTerm, selectedProviders, selectedModes, selectedFeatures]);

  const getUniqueProviders = (data: ModelGroupInfo[]) => {
    const providers = new Set<string>();
    data.forEach(model => {
      model.providers.forEach(provider => providers.add(provider));
    });
    return Array.from(providers);
  };

  const getUniqueModes = (data: ModelGroupInfo[]) => {
    const modes = new Set<string>();
    data.forEach(model => {
      if (model.mode) modes.add(model.mode);
    });
    return Array.from(modes);
  };

  const getUniqueFeatures = (data: ModelGroupInfo[]) => {
    const features = new Set<string>();
    data.forEach(model => {
      // Find all properties that start with 'supports_' and are true
      Object.entries(model)
        .filter(([key, value]) => key.startsWith('supports_') && value === true)
        .forEach(([key]) => {
          // Format the feature name (remove 'supports_' prefix and convert to title case)
          const featureName = key
            .replace(/^supports_/, '')
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
          features.add(featureName);
        });
    });
    return Array.from(features).sort();
  };



  const filteredData = useMemo(() => {
    if (!modelHubData) return [];
    
    let searchResults = modelHubData;
    
    // Apply search if there's a search term
    if (searchTerm.trim()) {
      const lowercaseSearch = searchTerm.toLowerCase();
      const searchWords = lowercaseSearch.split(/\s+/);
      
      // First, try flexible matching that handles different separators
      const exactMatches = modelHubData.filter(model => {
        const modelName = model.model_group.toLowerCase();
        
        // Check if it contains the exact search term
        if (modelName.includes(lowercaseSearch)) {
          return true;
        }
        
        // Check if it contains all search words (handles spaces vs slashes/dashes)
        return searchWords.every(word => modelName.includes(word));
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
          
          const aContainsWords = lowercaseSearch.split(/\s+/).every(word => aName.includes(word)) ? 50 : 0;
          const bContainsWords = lowercaseSearch.split(/\s+/).every(word => bName.includes(word)) ? 50 : 0;
          
          const aLength = aName.length;
          const bLength = bName.length;
          
          const aScore = aExactMatch + aStartsWith + aContainsWords + (1000 - aLength);
          const bScore = bExactMatch + bStartsWith + bContainsWords + (1000 - bLength);
          
                      return bScore - aScore; // Higher score first
          });
        }
    }
    
    // Apply other filters
    return searchResults.filter(model => {
      const matchesProvider = selectedProviders.length === 0 || selectedProviders.some(provider => model.providers.includes(provider));
      const matchesMode = selectedModes.length === 0 || selectedModes.includes(model.mode || "");
      
      // Check if model has any of the selected features
      const matchesFeature = selectedFeatures.length === 0 || 
        Object.entries(model)
          .filter(([key, value]) => key.startsWith('supports_') && value === true)
          .some(([key]) => {
            const featureName = key
              .replace(/^supports_/, '')
              .split('_')
              .map(word => word.charAt(0).toUpperCase() + word.slice(1))
              .join(' ');
            return selectedFeatures.includes(featureName);
          });
      
      return matchesProvider && matchesMode && matchesFeature;
    });
  }, [modelHubData, searchTerm, selectedProviders, selectedModes, selectedFeatures]);



  const showModal = (model: ModelGroupInfo) => {
    setSelectedModel(model);
    setIsModalVisible(true);
  };

  const handleModalOk = () => {
    setIsModalVisible(false);
    setSelectedModel(null);
  };

  const handleModalCancel = () => {
    setIsModalVisible(false);
    setSelectedModel(null);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    NotificationsManager.success("Copied to clipboard!");
  };

  const formatCapabilityName = (key: string) => {
    return key
      .replace(/^supports_/, '')
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const getModelCapabilities = (model: ModelGroupInfo) => {
    return Object.entries(model)
      .filter(([key, value]) => key.startsWith('supports_') && value === true)
      .map(([key]) => key);
  };



  const formatCost = (cost: number) => {
    return `$${(cost * 1_000_000).toFixed(4)}`;
  };

  const formatTokens = (tokens: number | undefined) => {
    if (!tokens) return "N/A";
    if (tokens >= 1000) {
      return `${(tokens / 1000).toFixed(0)}K`;
    }
    return tokens.toString();
  };

  const formatLimits = (rpm?: number, tpm?: number) => {
    const limits = [];
    if (rpm) limits.push(`RPM: ${rpm.toLocaleString()}`);
    if (tpm) limits.push(`TPM: ${tpm.toLocaleString()}`);
    return limits.length > 0 ? limits.join(", ") : "N/A";
  };



  const publicModelHubColumns = (): ColumnDef<ModelGroupInfo>[] => [
    {
      header: "Model Name",
      accessorKey: "model_group",
      enableSorting: true,
      cell: ({ row }) => (
        <div className="overflow-hidden">
          <Tooltip title={row.original.model_group}>
            <Button 
              size="xs"
              variant="light"
              className="font-mono text-blue-500 bg-blue-50 hover:bg-blue-100 text-xs font-normal px-2 py-0.5 text-left"
              onClick={() => showModal(row.original)}
            >
              {row.original.model_group}
            </Button>
          </Tooltip>
        </div>
      ),
      size: 150,
    },
    {
      header: "Providers",
      accessorKey: "providers",
      enableSorting: true,
      cell: ({ row }) => {
        const providers = row.original.providers;
        
        return (
          <div className="flex flex-wrap gap-1">
            {providers.map((provider) => {
              const { logo } = getProviderLogoAndName(provider);
              return (
                <div
                  key={provider}
                  className="flex items-center space-x-1 px-2 py-1 bg-gray-100 rounded text-xs"
                >
                  {logo && (
                    <img 
                      src={logo} 
                      alt={provider} 
                      className="w-3 h-3 flex-shrink-0 object-contain"
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = 'none';
                      }}
                    />
                  )}
                  <span className="capitalize">{provider}</span>
                </div>
              );
            })}
          </div>
        );
      },
      size: 120,
    },
    {
      header: "Mode",
      accessorKey: "mode",
      enableSorting: true,
      cell: ({ row }) => {
        const mode = row.original.mode;
        const getModeIcon = (mode: string) => {
          switch (mode?.toLowerCase()) {
            case "chat":
              return "💬";
            case "rerank":
              return "🔄";
            case "embedding":
              return "📄";
            default:
              return "🤖";
          }
        };
        
        return (
          <div className="flex items-center space-x-2">
            <span>{getModeIcon(mode || "")}</span>
            <Text>{mode || "Chat"}</Text>
          </div>
        );
      },
      size: 100,
    },
    {
      header: "Max Input",
      accessorKey: "max_input_tokens",
      enableSorting: true,
      cell: ({ row }) => (
        <Text className="text-center">{formatTokens(row.original.max_input_tokens)}</Text>
      ),
      size: 100,
      meta: {
        className: "text-center",
      },
    },
    {
      header: "Max Output",
      accessorKey: "max_output_tokens",
      enableSorting: true,
      cell: ({ row }) => (
        <Text className="text-center">{formatTokens(row.original.max_output_tokens)}</Text>
      ),
      size: 100,
      meta: {
        className: "text-center",
      },
    },
    {
      header: "Input $/1M",
      accessorKey: "input_cost_per_token",
      enableSorting: true,
      cell: ({ row }) => {
        const cost = row.original.input_cost_per_token;
        return (
          <Text className="text-center">
            {cost ? formatCost(cost) : "Free"}
          </Text>
        );
      },
      size: 100,
      meta: {
        className: "text-center",
      },
    },
    {
      header: "Output $/1M",
      accessorKey: "output_cost_per_token",
      enableSorting: true,
      cell: ({ row }) => {
        const cost = row.original.output_cost_per_token;
        return (
          <Text className="text-center">
            {cost ? formatCost(cost) : "Free"}
          </Text>
        );
      },
      size: 100,
      meta: {
        className: "text-center",
      },
    },
    {
      header: "Features",
      accessorKey: "supports_vision",
      enableSorting: false,
      cell: ({ row }) => {
        const model = row.original;
        
        // Dynamically get all features that start with 'supports_' and are true
        const features = Object.entries(model)
          .filter(([key, value]) => key.startsWith('supports_') && value === true)
          .map(([key]) => formatCapabilityName(key));
        
        if (features.length === 0) {
          return <Text className="text-gray-400">-</Text>;
        }
        
        if (features.length === 1) {
          return (
            <div className="h-6 flex items-center">
              <Tag color="blue" className="text-xs">
                {features[0]}
              </Tag>
            </div>
          );
        }
        
        return (
          <div className="h-6 flex items-center space-x-1">
            <Tag color="blue" className="text-xs">
              {features[0]}
            </Tag>
            <Tooltip 
              title={
                <div className="space-y-1">
                  <div className="font-medium">All Features:</div>
                  {features.map((feature, index) => (
                    <div key={index} className="text-xs">• {feature}</div>
                  ))}
                </div>
              }
              trigger="click"
              placement="topLeft"
            >
              <span 
                className="text-xs text-blue-600 cursor-pointer hover:text-blue-800 hover:underline"
                onClick={(e) => e.stopPropagation()}
              >
                +{features.length - 1}
              </span>
            </Tooltip>
          </div>
        );
      },
      size: 120,
    },
    {
      header: "Limits",
      accessorKey: "rpm",
      enableSorting: true,
      cell: ({ row }) => {
        const model = row.original;
        return (
          <Text className="text-xs text-gray-600">
            {formatLimits(model.rpm, model.tpm)}
          </Text>
        );
      },
      size: 150,
    },
  ];

  return (
    <ThemeProvider accessToken={accessToken}>
      <div className="min-h-screen bg-white">
        {/* Navigation */}
        <Navbar 
          userID={null}
          userEmail={null}
          userRole={null}
          premiumUser={false}
          setProxySettings={setProxySettings}
          proxySettings={proxySettings}
          accessToken={accessToken || null}
          isPublicPage={true}
        />

      <div className="w-full px-8 py-12">
        {/* About Section */}
          <Card className="mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm">
            <Title className="text-2xl font-semibold mb-6 text-gray-900">About</Title>
            <p className="text-gray-700 mb-6 text-base leading-relaxed">{customDocsDescription ? customDocsDescription : "Proxy Server to call 100+ LLMs in the OpenAI format."}</p>
            <div className="flex items-center space-x-3 text-sm text-gray-600">
              <span className="flex items-center">
                <span className="w-4 h-4 mr-2">🔧</span>
                Built with litellm: v{litellmVersion}
              </span>
            </div>
          </Card>

        {/* Useful Links */}
        {usefulLinks && Object.keys(usefulLinks).length > 0 && (
          <Card className="mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm">
            <Title className="text-2xl font-semibold mb-6 text-gray-900">Useful Links</Title>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(usefulLinks || {}).map(([title, url]) => (
                <button
                  key={title}
                  onClick={() => window.open(url, '_blank')}
                  className="flex items-center space-x-3 text-blue-600 hover:text-blue-800 transition-colors p-3 rounded-lg hover:bg-blue-50 border border-gray-200"
                >
                  <ExternalLinkIcon className="w-4 h-4" />
                  <Text className="text-sm font-medium">{title}</Text>
                </button>
              ))}
            </div>
          </Card>
        )}

        {/* Health and Endpoint Status */}
        <Card className="mb-10 p-8 bg-white border border-gray-200 rounded-lg shadow-sm">
          <Title className="text-2xl font-semibold mb-6 text-gray-900">Health and Endpoint Status</Title>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Text className="text-green-600 font-medium text-sm">Service status: {serviceStatus}</Text>
          </div>
        </Card>

        {/* Models Table */}
        <Card className="p-8 bg-white border border-gray-200 rounded-lg shadow-sm">
          <div className="flex justify-between items-center mb-8">
            <Title className="text-2xl font-semibold text-gray-900">Available Models</Title>
          </div>

          {/* Filters */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8 p-6 bg-gray-50 rounded-lg border border-gray-200">
            <div>
              <div className="flex items-center space-x-2 mb-3">
                <Text className="text-sm font-medium text-gray-700">Search Models:</Text>
                <Tooltip title="Smart search with relevance ranking - finds models containing your search terms, ranked by relevance. Try searching 'xai grok-4', 'claude-4', 'gpt-4', or 'sonnet'" placement="top">
                  <Info className="w-4 h-4 text-gray-400 cursor-help" />
                </Tooltip>
              </div>
              <div className="relative">
                <SearchIcon className="w-4 h-4 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
                <input
                  type="text"
                  placeholder="Search model names... (smart search enabled)"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="border border-gray-300 rounded-lg pl-10 pr-4 py-2 w-full text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white"
                />
              </div>
            </div>
            <div>
              <Text className="text-sm font-medium mb-3 text-gray-700">Provider:</Text>
              <Select
                mode="multiple"
                value={selectedProviders}
                onChange={(values) => setSelectedProviders(values)}
                placeholder="Select providers"
                className="w-full"
                size="large"
                allowClear
                optionRender={(option) => {
                  const { logo } = getProviderLogoAndName(option.value as string);
                  return (
                    <div className="flex items-center space-x-2">
                      {logo && (
                        <img 
                          src={logo} 
                          alt={option.label as string} 
                          className="w-5 h-5 flex-shrink-0 object-contain"
                          onError={(e) => {
                            (e.target as HTMLImageElement).style.display = 'none';
                          }}
                        />
                      )}
                      <span className="capitalize">{option.label}</span>
                    </div>
                  );
                }}
              >
                {modelHubData && getUniqueProviders(modelHubData).map(provider => (
                  <Select.Option key={provider} value={provider}>
                    {provider}
                  </Select.Option>
                ))}
              </Select>
            </div>
            <div>
              <Text className="text-sm font-medium mb-3 text-gray-700">Mode:</Text>
              <Select
                mode="multiple"
                value={selectedModes}
                onChange={(values) => setSelectedModes(values)}
                placeholder="Select modes"
                className="w-full"
                size="large"
                allowClear
              >
                {modelHubData && getUniqueModes(modelHubData).map(mode => (
                  <Select.Option key={mode} value={mode}>{mode}</Select.Option>
                ))}
              </Select>
            </div>
            <div>
              <Text className="text-sm font-medium mb-3 text-gray-700">Features:</Text>
              <Select
                mode="multiple"
                value={selectedFeatures}
                onChange={(values) => setSelectedFeatures(values)}
                placeholder="Select features"
                className="w-full"
                size="large"
                allowClear
              >
                {modelHubData && getUniqueFeatures(modelHubData).map(feature => (
                  <Select.Option key={feature} value={feature}>{feature}</Select.Option>
                ))}
              </Select>
            </div>
          </div>

          <ModelDataTable
            columns={publicModelHubColumns()}
            data={filteredData}
            isLoading={loading}
            table={tableRef}
            defaultSorting={[{ id: "model_group", desc: false }]}
          />

          <div className="mt-8 text-center">
            <Text className="text-sm text-gray-600">
              Showing {filteredData.length} of {modelHubData?.length || 0} models
            </Text>
          </div>
        </Card>
      </div>

      {/* Model Details Modal */}
      <Modal
        title={
          <div className="flex items-center space-x-2">
            <span>{selectedModel?.model_group || "Model Details"}</span>
            {selectedModel && (
              <Tooltip title="Copy model name">
                <Copy
                  onClick={() => copyToClipboard(selectedModel.model_group)}
                  className="cursor-pointer text-gray-500 hover:text-blue-500 w-4 h-4"
                />
              </Tooltip>
            )}
          </div>
        }
        width={1000}
        open={isModalVisible}
        footer={null}
        onOk={handleModalOk}
        onCancel={handleModalCancel}
      >
        {selectedModel && (
          <div className="space-y-6">
            {/* Model Overview */}
            <div>
              <Text className="text-lg font-semibold mb-4">Model Overview</Text>
              <div className="grid grid-cols-2 gap-4 mb-4">
                <div>
                  <Text className="font-medium">Model Name:</Text>
                  <Text>{selectedModel.model_group}</Text>
                </div>
                <div>
                  <Text className="font-medium">Mode:</Text>
                  <Text>{selectedModel.mode || "Not specified"}</Text>
                </div>
                <div>
                  <Text className="font-medium">Providers:</Text>
                  <div className="flex flex-wrap gap-1 mt-1">
                    {selectedModel.providers.map(provider => {
                      const { logo } = getProviderLogoAndName(provider);
                      return (
                        <Tag key={provider} color="blue">
                          <div className="flex items-center space-x-1">
                            {logo && (
                              <img 
                                src={logo} 
                                alt={provider} 
                                className="w-3 h-3 flex-shrink-0 object-contain"
                                onError={(e) => {
                                  (e.target as HTMLImageElement).style.display = 'none';
                                }}
                              />
                            )}
                            <span className="capitalize">{provider}</span>
                          </div>
                        </Tag>
                      );
                    })}
                  </div>
                </div>
              </div>
              
              {/* Wildcard Routing Note */}
              {selectedModel.model_group.includes('*') && (
                <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                  <div className="flex items-start space-x-2">
                    <Info className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                    <div>
                      <Text className="font-medium text-blue-900 mb-2">Wildcard Routing</Text>
                      <Text className="text-sm text-blue-800 mb-2">
                        This model uses wildcard routing. You can pass any value where you see the <code className="bg-blue-100 px-1 py-0.5 rounded text-xs">*</code> symbol.
                      </Text>
                      <Text className="text-sm text-blue-800">
                        For example, with <code className="bg-blue-100 px-1 py-0.5 rounded text-xs">{selectedModel.model_group}</code>, you can use any string (<code className="bg-blue-100 px-1 py-0.5 rounded text-xs">{selectedModel.model_group.replace('*', 'my-custom-value')}</code>) that matches this pattern.
                      </Text>
                    </div>
                  </div>
                </div>
              )}
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
                  <Text>{selectedModel.input_cost_per_token ? formatCost(selectedModel.input_cost_per_token) : "Not specified"}</Text>
                </div>
                <div>
                  <Text className="font-medium">Output Cost per 1M Tokens:</Text>
                  <Text>{selectedModel.output_cost_per_token ? formatCost(selectedModel.output_cost_per_token) : "Not specified"}</Text>
                </div>
              </div>
            </div>

            {/* Capabilities */}
            <div>
              <Text className="text-lg font-semibold mb-4">Capabilities</Text>
              <div className="flex flex-wrap gap-2">
                {(() => {
                  const capabilities = getModelCapabilities(selectedModel);
                  const colors = ['green', 'blue', 'purple', 'orange', 'red', 'yellow'];
                  
                  if (capabilities.length === 0) {
                    return <Text className="text-gray-500">No special capabilities listed</Text>;
                  }
                  
                  return capabilities.map((capability, index) => (
                    <Tag 
                      key={capability} 
                      color={colors[index % colors.length]}
                    >
                      {formatCapabilityName(capability)}
                    </Tag>
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
                  {selectedModel.supported_openai_params.map(param => (
                    <Tag key={param} color="green">{param}</Tag>
                  ))}
                </div>
              </div>
            )}

            {/* Usage Example */}
            <div>
              <Text className="text-lg font-semibold mb-4">Usage Example</Text>
              <div className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto">
                <pre className="text-sm">
{(() => {
  const codeSnippet = generateCodeSnippet({
    apiKeySource: 'custom',
    accessToken: null,
    apiKey: 'your_api_key',
    inputMessage: 'Hello, how are you?',
    chatHistory: [
      { role: 'user', content: 'Hello, how are you?', isImage: false } as MessageType
    ],
    selectedTags: [],
    selectedVectorStores: [],
    selectedGuardrails: [],
    selectedMCPTools: [],
    endpointType: getEndpointType(selectedModel.mode || 'chat'),
    selectedModel: selectedModel.model_group,
    selectedSdk: 'openai'
  });
  return codeSnippet;
})()}
                </pre>
              </div>
              <div className="mt-2 text-right">
                                  <button
                    onClick={() => {
                      const codeSnippet = generateCodeSnippet({
                        apiKeySource: 'custom',
                        accessToken: null,
                        apiKey: 'your_api_key',
                        inputMessage: 'Hello, how are you?',
                        chatHistory: [
                          { role: 'user', content: 'Hello, how are you?', isImage: false } as MessageType
                        ],
                        selectedTags: [],
                        selectedVectorStores: [],
                        selectedGuardrails: [],
                        selectedMCPTools: [],
                        endpointType: getEndpointType(selectedModel.mode || 'chat'),
                        selectedModel: selectedModel.model_group,
                        selectedSdk: 'openai'
                      });
                      copyToClipboard(codeSnippet);
                    }}
                    className="text-sm text-blue-600 hover:text-blue-800 cursor-pointer"
                  >
                  Copy to clipboard
                </button>
              </div>
            </div>
          </div>
        )}
      </Modal>
      </div>
    </ThemeProvider>
  );
};

export default PublicModelHub; 