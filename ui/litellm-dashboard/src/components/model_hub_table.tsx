import React, { useEffect, useState, useRef, useCallback } from "react";
import { useRouter } from "next/navigation";
import { modelHubCall, modelHubPublicModelsCall, getProxyBaseUrl } from "./networking";
import { getConfigFieldSetting } from "./networking";
import { ModelDataTable } from "./model_dashboard/table";
import { modelHubColumns } from "./model_hub_table_columns";
import PublicModelHub from "./public_model_hub";
import MakeModelPublicForm from "./make_model_public_form";
import ModelFilters from "./model_filters";
import UsefulLinksManagement from "./useful_links_management";
import { Card, Text, Title, Button, Badge } from "@tremor/react";
import { Modal } from "antd";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { Table as TableInstance } from "@tanstack/react-table";
import { Copy } from "lucide-react";
import { isAdminRole } from "../utils/roles";
import NotificationsManager from "./molecules/notifications_manager";

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
    };

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

  const handleMakePublicPage = () => {
    if (!accessToken) {
      return;
    }

    // Show the modal for selecting models to make public
    setIsMakePublicModalVisible(true);
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
          <div className="flex justify-between items-center mb-6">
            <div className="flex flex-col items-start">
              <Title className="text-center">Model Hub</Title>
              {isAdminRole(userRole || "") ? (
                <p className="text-sm text-gray-600">
                  Make models public for developers to know what models are available on the proxy.
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

              {publicPage == false && isAdminRole(userRole || "") && (
                <Button className="ml-4" onClick={() => handleMakePublicPage()}>
                  Make Public
                </Button>
              )}
            </div>
          </div>

          {/* Useful Links Management Section for Admins */}
          {isAdminRole(userRole || "") && (
            <div className="mt-8 mb-2">
              <UsefulLinksManagement accessToken={accessToken} userRole={userRole} />
            </div>
          )}

          {/* Model Filters and Table */}
          <Card>
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

      {/* Make Model Public Form */}
      <MakeModelPublicForm
        visible={isMakePublicModalVisible}
        onClose={() => setIsMakePublicModalVisible(false)}
        accessToken={accessToken || ""}
        modelHubData={modelHubData || []}
        onSuccess={handleMakePublicSuccess}
      />
    </div>
  );
};

export default ModelHubTable;
