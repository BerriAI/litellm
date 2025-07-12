import React, { useEffect, useState, useRef } from "react";
import { modelHubPublicModelsCall, proxyBaseUrl, getUiConfig, getPublicModelHubInfo } from "./networking";
import { ModelDataTable } from "./model_dashboard/table";
import { ColumnDef } from "@tanstack/react-table";
import {
  Card,
  Text,
  Title,
  Button,
} from "@tremor/react";
import { message, Tag, Tooltip, Modal } from "antd";
import { CopyOutlined } from "@ant-design/icons";
import { ExternalLinkIcon, SearchIcon, EyeIcon, CogIcon } from "@heroicons/react/outline";
import { Copy, Info } from "lucide-react";
import { Table as TableInstance } from '@tanstack/react-table';
import { generateCodeSnippet } from "./chat_ui/CodeSnippets";
import { EndpointType, getEndpointType } from "./chat_ui/mode_endpoint_mapping";
import { MessageType } from "./chat_ui/types";
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
  const [selectedProvider, setSelectedProvider] = useState<string>("");
  const [selectedMode, setSelectedMode] = useState<string>("");
  const [selectedFeature, setSelectedFeature] = useState<string>("");
  const [serviceStatus, setServiceStatus] = useState<string>("I'm alive! ✓");
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [selectedModel, setSelectedModel] = useState<null | ModelGroupInfo>(null);
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
  }, [searchTerm, selectedProvider, selectedMode, selectedFeature]);

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

  const filteredData = modelHubData?.filter(model => {
    const matchesSearch = model.model_group.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesProvider = selectedProvider === "" || model.providers.includes(selectedProvider);
    const matchesMode = selectedMode === "" || model.mode === selectedMode;
    
    // Check if model has the selected feature
    const matchesFeature = selectedFeature === "" || 
      Object.entries(model)
        .filter(([key, value]) => key.startsWith('supports_') && value === true)
        .some(([key]) => {
          const featureName = key
            .replace(/^supports_/, '')
            .split('_')
            .map(word => word.charAt(0).toUpperCase() + word.slice(1))
            .join(' ');
          return featureName === selectedFeature;
        });
    
    return matchesSearch && matchesProvider && matchesMode && matchesFeature;
  }) || [];



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
    message.success("Copied to clipboard!");
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
      header: "#",
      id: "index",
      enableSorting: false,
      cell: ({ row }) => {
        const index = row.index + 1;
        return <Text className="text-center">{index}</Text>;
      },
      size: 50,
    },
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
        const getProviderColor = (provider: string) => {
          switch (provider.toLowerCase()) {
            case "openai":
              return "green";
            case "anthropic":
              return "orange";
            case "cohere":
              return "blue";
            default:
              return "gray";
          }
        };
        
        return (
          <div className="flex flex-wrap gap-1">
            {providers.map((provider) => (
              <Tag
                key={provider}
                color={getProviderColor(provider)}
                className="text-xs"
              >
                {provider}
              </Tag>
            ))}
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
    },
    {
      header: "Max Output",
      accessorKey: "max_output_tokens",
      enableSorting: true,
      cell: ({ row }) => (
        <Text className="text-center">{formatTokens(row.original.max_output_tokens)}</Text>
      ),
      size: 100,
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
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-green-600 text-white px-8 py-6">
        <div className="flex justify-between items-center w-full">
          <Title className="text-white text-2xl font-semibold">{pageTitle}</Title>
        </div>
      </div>

      <div className="w-full px-8 py-12">
        {/* About Section */}
          <Card className="mb-10 p-8">
            <Title className="text-3xl font-semibold mb-6">About</Title>
            <p className="text-gray-700 mb-6 text-lg leading-relaxed">{customDocsDescription ? customDocsDescription : "Proxy Server to call 100+ LLMs in the OpenAI format."}</p>
            <div className="flex items-center space-x-3 text-base text-gray-600">
              <span className="flex items-center">
                <span className="w-5 h-5 mr-2">🔧</span>
                Built with litellm: v{litellmVersion}
              </span>
            </div>
          </Card>

        {/* Useful Links */}
        {usefulLinks && Object.keys(usefulLinks).length > 0 && (
          <Card className="mb-10 p-8">
            <Title className="text-3xl font-semibold mb-6">Useful Links</Title>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
              {Object.entries(usefulLinks || {}).map(([title, url]) => (
                <button
                  key={title}
                  onClick={() => window.open(url, '_blank')}
                  className="flex items-center space-x-3 text-blue-600 hover:text-blue-800 transition-colors p-4 rounded-lg hover:bg-blue-50"
                >
                  <ExternalLinkIcon className="w-5 h-5" />
                  <Text className="text-base font-medium">{title}</Text>
                </button>
              ))}
            </div>
          </Card>
        )}

        {/* Health and Endpoint Status */}
        <Card className="mb-10 p-8">
          <Title className="text-3xl font-semibold mb-6">Health and Endpoint Status</Title>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <Text className="text-green-600 font-medium text-base">Service status: {serviceStatus}</Text>
          </div>
        </Card>

        {/* Filters */}
        <Card className="mb-10 p-8">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            <div>
              <Text className="text-base font-medium mb-3">Search Models:</Text>
              <div className="relative">
                <SearchIcon className="w-5 h-5 text-gray-400 absolute left-3 top-1/2 transform -translate-y-1/2" />
                <input
                  type="text"
                  placeholder="Search model names..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="border rounded-lg pl-10 pr-4 py-3 w-full text-base focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
              </div>
            </div>
            <div>
              <Text className="text-base font-medium mb-3">Provider:</Text>
              <select
                value={selectedProvider}
                onChange={(e) => setSelectedProvider(e.target.value)}
                className="border rounded-lg px-4 py-3 text-base text-gray-600 w-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="" className="text-base text-gray-600">All Providers</option>
                {modelHubData && getUniqueProviders(modelHubData).map(provider => (
                  <option key={provider} value={provider} className="text-base text-gray-800">{provider}</option>
                ))}
              </select>
            </div>
            <div>
              <Text className="text-base font-medium mb-3">Mode:</Text>
              <select
                value={selectedMode}
                onChange={(e) => setSelectedMode(e.target.value)}
                className="border rounded-lg px-4 py-3 text-base text-gray-600 w-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="" className="text-base text-gray-600">All Modes</option>
                {modelHubData && getUniqueModes(modelHubData).map(mode => (
                  <option key={mode} value={mode} className="text-base text-gray-800">{mode}</option>
                ))}
              </select>
            </div>
            <div>
              <Text className="text-base font-medium mb-3">Features:</Text>
              <select
                value={selectedFeature}
                onChange={(e) => setSelectedFeature(e.target.value)}
                className="border rounded-lg px-4 py-3 text-base text-gray-600 w-full focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="" className="text-base text-gray-600">All Features</option>
                {modelHubData && getUniqueFeatures(modelHubData).map(feature => (
                  <option key={feature} value={feature} className="text-base text-gray-800">{feature}</option>
                ))}
              </select>
            </div>
          </div>
        </Card>

        {/* Models Table */}
        <Card className="p-8">
          <div className="flex justify-between items-center mb-8">
            <Title className="text-3xl font-semibold">Models available in Gateway</Title>
          </div>

          <ModelDataTable
            columns={publicModelHubColumns()}
            data={filteredData}
            isLoading={loading}
            table={tableRef}
            defaultSorting={[{ id: "model_group", desc: false }]}
          />

          <div className="mt-8 text-center">
            <Text className="text-base text-gray-600">
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
                    {selectedModel.providers.map(provider => (
                      <Tag key={provider} color="blue">{provider}</Tag>
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
  );
};

export default PublicModelHub; 