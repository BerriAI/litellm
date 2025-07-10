import React, { useEffect, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { modelHubCall } from "./networking";
import { getConfigFieldSetting, updateConfigFieldSetting } from "./networking";
import {
  Card,
  Text,
  Title,
  Button,
  Badge,
  Table,
  TableHead,
  TableHeaderCell,
  TableBody,
  TableRow,
  TableCell,
  Flex,
} from "@tremor/react";
import { 
  CopyOutlined, 
  InfoCircleOutlined
} from "@ant-design/icons";
import { Modal, Tooltip, message, Tag } from "antd";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";

interface ModelHubTableProps {
  accessToken: string | null;
  publicPage: boolean;
  premiumUser: boolean;
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
}

const ModelHubTable: React.FC<ModelHubTableProps> = ({
  accessToken,
  publicPage,
  premiumUser,
}) => {
  const [publicPageAllowed, setPublicPageAllowed] = useState<boolean>(false);
  const [modelHubData, setModelHubData] = useState<ModelGroupInfo[] | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [isPublicPageModalVisible, setIsPublicPageModalVisible] = useState(false);
  const [selectedModel, setSelectedModel] = useState<null | ModelGroupInfo>(null);
  const [searchTerm, setSearchTerm] = useState<string>("");
  const [selectedProvider, setSelectedProvider] = useState<string>("");
  const [selectedMode, setSelectedMode] = useState<string>("");
  const router = useRouter();

  useEffect(() => {
    if (!accessToken) {
      return;
    }

    const fetchData = async () => {
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

    fetchData();
  }, [accessToken, publicPage]);

  const showModal = (model: ModelGroupInfo) => {
    setSelectedModel(model);
    setIsModalVisible(true);
  };

  const goToPublicModelPage = () => {
    router.replace(`/model_hub_table?key=${accessToken}`);
  };

  const handleMakePublicPage = async () => {
    if (!accessToken) {
      return;
    }
    updateConfigFieldSetting(accessToken, "enable_public_model_hub", true).then(
      (data) => {
        setIsPublicPageModalVisible(true);
      }
    );
  };

  const handleOk = () => {
    setIsModalVisible(false);
    setIsPublicPageModalVisible(false);
    setSelectedModel(null);
  };

  const handleCancel = () => {
    setIsModalVisible(false);
    setIsPublicPageModalVisible(false);
    setSelectedModel(null);
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    message.success("Copied to clipboard!");
  };

  const formatCost = (cost: number) => {
    return `$${(cost * 1_000_000).toFixed(2)}`;
  };

  const formatTokens = (tokens: number) => {
    if (tokens >= 1_000_000) {
      return `${(tokens / 1_000_000).toFixed(1)}M`;
    } else if (tokens >= 1_000) {
      return `${(tokens / 1_000).toFixed(1)}K`;
    }
    return tokens.toString();
  };

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

  const filteredData = modelHubData?.filter(model => {
    const matchesSearch = model.model_group.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesProvider = selectedProvider === "" || model.providers.includes(selectedProvider);
    const matchesMode = selectedMode === "" || model.mode === selectedMode;
    return matchesSearch && matchesProvider && matchesMode;
  }) || [];



  return (
    <div>
      {(publicPage && publicPageAllowed) || publicPage == false ? (
        <div className="w-full m-2 mt-2 p-8">
          <div className="flex justify-between items-center mb-6">
            <Title className="text-center">Model Hub - Table View</Title>
            {publicPage == false ? (
              premiumUser ? (
                <Button className="ml-4" onClick={() => handleMakePublicPage()}>
                  ✨ Make Public
                </Button>
              ) : (
                <Button className="ml-4">
                  <a href="https://forms.gle/W3U4PZpJGFHWtHyA9" target="_blank">
                    ✨ Make Public
                  </a>
                </Button>
              )
            ) : (
              <div className="flex items-center space-x-4">
                <Text>Filter by key:</Text>
                <Text className="bg-gray-200 px-2 py-1 rounded">{`/ui/model_hub_table?key=<YOUR_KEY>`}</Text>
              </div>
            )}
          </div>

          {/* Filters */}
          <Card className="mb-6">
            <div className="flex flex-wrap gap-4 items-center">
              <div>
                <Text className="text-sm font-medium mb-2">Search Models:</Text>
                <input
                  type="text"
                  placeholder="Search model names..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="border rounded px-3 py-2 w-64"
                />
              </div>
              <div>
                <Text className="text-sm font-medium mb-2">Provider:</Text>
                <select
                  value={selectedProvider}
                  onChange={(e) => setSelectedProvider(e.target.value)}
                  className="border rounded px-3 py-2"
                >
                  <option value="">All Providers</option>
                  {modelHubData && getUniqueProviders(modelHubData).map(provider => (
                    <option key={provider} value={provider}>{provider}</option>
                  ))}
                </select>
              </div>
              <div>
                <Text className="text-sm font-medium mb-2">Mode:</Text>
                <select
                  value={selectedMode}
                  onChange={(e) => setSelectedMode(e.target.value)}
                  className="border rounded px-3 py-2"
                >
                  <option value="">All Modes</option>
                  {modelHubData && getUniqueModes(modelHubData).map(mode => (
                    <option key={mode} value={mode}>{mode}</option>
                  ))}
                </select>
              </div>
            </div>
          </Card>

          {/* Model Table */}
          <Card>
            {loading ? (
              <div className="text-center py-8">
                <Text>Loading models...</Text>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <Table className="min-w-full">
                <TableHead>
                  <TableRow>
                    <TableHeaderCell className="min-w-40">Model</TableHeaderCell>
                    <TableHeaderCell className="min-w-20 hidden md:table-cell">Provider</TableHeaderCell>
                    <TableHeaderCell className="min-w-16 hidden lg:table-cell">Mode</TableHeaderCell>
                    <TableHeaderCell className="min-w-24 hidden lg:table-cell">Tokens</TableHeaderCell>
                    <TableHeaderCell className="min-w-24">Cost/1M</TableHeaderCell>
                    <TableHeaderCell className="min-w-28">Features</TableHeaderCell>
                    <TableHeaderCell className="min-w-16">Details</TableHeaderCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {filteredData.map((model) => (
                    <TableRow key={model.model_group}>
                      <TableCell>
                        <div className="space-y-1">
                          <div className="flex items-center space-x-2">
                            <Text className="font-medium text-sm">{model.model_group}</Text>
                            <Tooltip title="Copy model name">
                              <CopyOutlined
                                onClick={() => copyToClipboard(model.model_group)}
                                className="cursor-pointer text-gray-500 hover:text-blue-500 text-xs"
                              />
                            </Tooltip>
                          </div>
                          {/* Show provider on mobile when provider column is hidden */}
                          <div className="md:hidden">
                            <Text className="text-xs text-gray-600">
                              {model.providers.join(", ")}
                            </Text>
                          </div>
                        </div>
                      </TableCell>
                      <TableCell className="hidden md:table-cell">
                        <div className="flex flex-wrap gap-1">
                          {model.providers.slice(0, 2).map(provider => (
                            <Tag key={provider} color="blue" className="text-xs">
                              {provider}
                            </Tag>
                          ))}
                          {model.providers.length > 2 && (
                            <Text className="text-xs text-gray-500">+{model.providers.length - 2}</Text>
                          )}
                        </div>
                      </TableCell>
                      <TableCell className="hidden lg:table-cell">
                        {model.mode ? (
                          <Badge color="green" size="sm">{model.mode}</Badge>
                        ) : (
                          <Text className="text-gray-500">-</Text>
                        )}
                      </TableCell>
                      <TableCell className="hidden lg:table-cell">
                        <div className="space-y-1">
                          <Text className="text-xs">
                            {model.max_input_tokens ? formatTokens(model.max_input_tokens) : "-"} / {model.max_output_tokens ? formatTokens(model.max_output_tokens) : "-"}
                          </Text>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="space-y-1">
                          <Text className="text-xs">
                            {model.input_cost_per_token ? formatCost(model.input_cost_per_token) : "-"}
                          </Text>
                          <Text className="text-xs text-gray-500">
                            {model.output_cost_per_token ? formatCost(model.output_cost_per_token) : "-"}
                          </Text>
                        </div>
                      </TableCell>
                      <TableCell>
                        <div className="flex flex-wrap gap-1">
                          {model.supports_function_calling && (
                            <Badge color="green" size="xs">Fn</Badge>
                          )}
                          {model.supports_vision && (
                            <Badge color="blue" size="xs">Vision</Badge>
                          )}
                          {model.supports_parallel_function_calling && (
                            <Badge color="purple" size="xs">Parallel</Badge>
                          )}
                          {!model.supports_function_calling && !model.supports_vision && !model.supports_parallel_function_calling && (
                            <Text className="text-gray-500 text-xs">-</Text>
                          )}
                        </div>
                      </TableCell>
                      <TableCell>
                        <Button
                          size="xs"
                          variant="secondary"
                          onClick={() => showModal(model)}
                          icon={InfoCircleOutlined}
                        >
                          <span className="hidden lg:inline">Details</span>
                          <span className="lg:hidden">Info</span>
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
                </Table>
              </div>
            )}
          </Card>

          <div className="mt-4 text-center">
            <Text className="text-sm text-gray-600">
              Showing {filteredData.length} of {modelHubData?.length || 0} models
            </Text>
          </div>
        </div>
      ) : (
        <Card className="mx-auto max-w-xl mt-10">
          <Text className="text-xl text-center mb-2 text-black">
            Public Model Hub not enabled.
          </Text>
          <p className="text-base text-center text-slate-800">
            Ask your proxy admin to enable this on their Admin UI.
          </p>
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
              {`<proxy_base_url>/ui/model_hub_table?key=<YOUR_API_KEY>`}
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
                {selectedModel.supports_function_calling && (
                  <Badge color="green">Function Calling</Badge>
                )}
                {selectedModel.supports_vision && (
                  <Badge color="blue">Vision Support</Badge>
                )}
                {selectedModel.supports_parallel_function_calling && (
                  <Badge color="purple">Parallel Function Calling</Badge>
                )}
                {!selectedModel.supports_function_calling && !selectedModel.supports_vision && !selectedModel.supports_parallel_function_calling && (
                  <Text className="text-gray-500">No special capabilities listed</Text>
                )}
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
    </div>
  );
};

export default ModelHubTable; 