import React, { useEffect, useState, useRef } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import { modelHubCall, makeModelGroupPublic, modelHubPublicModelsCall, proxyBaseUrl } from "./networking";
import { getConfigFieldSetting, updateConfigFieldSetting } from "./networking";
import { ModelDataTable } from "./model_dashboard/table";
import { modelHubColumns } from "./model_hub_table_columns";
import PublicModelHub from "./public_model_hub";
import {
  Card,
  Text,
  Title,
  Button,
  Badge,
  Flex,
} from "@tremor/react";
import { Modal, message } from "antd";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { Table as TableInstance } from '@tanstack/react-table';

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
  public?: boolean; // Whether the model is public (defaults to false)
  // Allow any additional properties for flexibility
  [key: string]: any;
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
  const [selectedFeature, setSelectedFeature] = useState<string>("");
  const [selectedModels, setSelectedModels] = useState<Set<string>>(new Set());
  const router = useRouter();
  const tableRef = useRef<TableInstance<any>>(null);

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
    }

    if (accessToken) {
      fetchData(accessToken);
    } else if (publicPage) {
      fetchPublicData();
    }
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
    
    try {
      // Get the selected model groups or use all if none are selected
      const modelGroupsToMakePublic = selectedModels.size > 0 
        ? Array.from(selectedModels)
        : modelHubData?.map(model => model.model_group) || [];
      
      if (modelGroupsToMakePublic.length > 0) {
        // Call the endpoint to make the selected model groups public
        await makeModelGroupPublic(accessToken, modelGroupsToMakePublic);
        
        // Show success message
        message.success(`Successfully made ${modelGroupsToMakePublic.length} model group(s) public!`);
        
        // Route to the model hub table
        router.push(`/ui/model_hub_table`);
      } else {
        // Show the modal if no model groups available
        setIsPublicPageModalVisible(true);
      }
    } catch (error) {
      console.error("Error making model groups public:", error);
      message.error("Failed to make model groups public. Please try again.");
    }
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

  const formatCapabilityName = (key: string) => {
    // Remove 'supports_' prefix and convert snake_case to Title Case
    return key
      .replace(/^supports_/, '')
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  const getModelCapabilities = (model: ModelGroupInfo) => {
    // Find all properties that start with 'supports_' and are true
    return Object.entries(model)
      .filter(([key, value]) => key.startsWith('supports_') && value === true)
      .map(([key]) => key);
  };

  const formatCost = (cost: number) => {
    return `$${(cost * 1_000_000).toFixed(2)}`;
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

  const handleRowSelection = (modelGroup: string, isSelected: boolean) => {
    const newSelection = new Set(selectedModels);
    if (isSelected) {
      newSelection.add(modelGroup);
    } else {
      newSelection.delete(modelGroup);
    }
    setSelectedModels(newSelection);
  };

  const handleSelectAll = (checked: boolean) => {
    console.log("checked", checked);
    if (checked) {
      const allModelGroups = filteredData.map(model => model.model_group);
      setSelectedModels(new Set(allModelGroups));
    } else {
      setSelectedModels(new Set());
    }
  };

  // Use the same logic as health check columns
  const allModelsSelected = filteredData.length > 0 && filteredData.every(model => selectedModels.has(model.model_group));
  const isIndeterminate = selectedModels.size > 0 && !allModelsSelected;

  // Clear selections when filters change to avoid confusion
  useEffect(() => {
    setSelectedModels(new Set());
  }, [searchTerm, selectedProvider, selectedMode, selectedFeature]);

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
                  className="border rounded px-3 py-2 w-64 h-10 text-sm"
                />
              </div>
              <div>
                <Text className="text-sm font-medium mb-2">Provider:</Text>
                <select
                  value={selectedProvider}
                  onChange={(e) => setSelectedProvider(e.target.value)}
                  className="border rounded px-3 py-2 text-sm text-gray-600 w-40 h-10"
                >
                  <option value="" className="text-sm text-gray-600">All Providers</option>
                  {modelHubData && getUniqueProviders(modelHubData).map(provider => (
                    <option key={provider} value={provider} className="text-sm text-gray-800">{provider}</option>
                  ))}
                </select>
              </div>
              <div>
                <Text className="text-sm font-medium mb-2">Mode:</Text>
                <select
                  value={selectedMode}
                  onChange={(e) => setSelectedMode(e.target.value)}
                  className="border rounded px-3 py-2 text-sm text-gray-600 w-32 h-10"
                >
                  <option value="" className="text-sm text-gray-600">All Modes</option>
                  {modelHubData && getUniqueModes(modelHubData).map(mode => (
                    <option key={mode} value={mode} className="text-sm text-gray-800">{mode}</option>
                  ))}
                </select>
              </div>
              <div>
                <Text className="text-sm font-medium mb-2">Features:</Text>
                <select
                  value={selectedFeature}
                  onChange={(e) => setSelectedFeature(e.target.value)}
                  className="border rounded px-3 py-2 text-sm text-gray-600 w-48 h-10"
                >
                  <option value="" className="text-sm text-gray-600">All Features</option>
                  {modelHubData && getUniqueFeatures(modelHubData).map(feature => (
                    <option key={feature} value={feature} className="text-sm text-gray-800">{feature}</option>
                  ))}
                </select>
              </div>
            </div>
          </Card>

          {/* Model Table */}
          <ModelDataTable
            columns={modelHubColumns(
              selectedModels,
              allModelsSelected,
              isIndeterminate,
              handleRowSelection,
              handleSelectAll,
              showModal,
              copyToClipboard,
              publicPage,
            )}
            data={filteredData}
            isLoading={loading}
            table={tableRef}
            defaultSorting={[{ id: "model_group", desc: false }]}
          />

          <div className="mt-4 text-center space-y-2">
            <Text className="text-sm text-gray-600">
              Showing {filteredData.length} of {modelHubData?.length || 0} models
            </Text>
            {selectedModels.size > 0 && (
              <div className="flex items-center justify-center space-x-4">
                <Text className="text-sm text-blue-600">
                  {selectedModels.size} model{selectedModels.size !== 1 ? 's' : ''} selected
                </Text>
                <Button
                  size="xs"
                  variant="secondary"
                  onClick={() => setSelectedModels(new Set())}
                  className="text-xs"
                >
                  Unselect All
                </Button>
              </div>
            )}
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
              {`${proxyBaseUrl}/model_hub_table`}
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
                      <Badge key={provider} color="blue">{provider}</Badge>
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
                    <Badge 
                      key={capability} 
                      color={colors[index % colors.length]}
                    >
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
                  {selectedModel.supported_openai_params.map(param => (
                    <Badge key={param} color="green">{param}</Badge>
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