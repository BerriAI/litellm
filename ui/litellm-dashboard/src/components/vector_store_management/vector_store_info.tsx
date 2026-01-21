import React, { useState, useEffect } from "react";
import { Card, Text, Title, Button, Badge, TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import { Form, Input, Select as Select2, Tooltip, Button as AntButton } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { ArrowLeftIcon } from "@heroicons/react/outline";
import {
  vectorStoreInfoCall,
  vectorStoreUpdateCall,
  credentialListCall,
  CredentialItem,
  detectEmbeddingDimensionCall,
} from "../networking";
import { VectorStore } from "./types";
import {
  VectorStoreProviders,
  vectorStoreProviderLogoMap,
  vectorStoreProviderMap,
  getProviderSpecificFields,
  VectorStoreFieldConfig,
} from "../vector_store_providers";
import VectorStoreTester from "./VectorStoreTester";
import NotificationsManager from "../molecules/notifications_manager";
import { fetchAvailableModels, ModelGroup } from "../playground/llm_calls/fetch_models";

interface VectorStoreInfoViewProps {
  vectorStoreId: string;
  onClose: () => void;
  accessToken: string | null;
  is_admin: boolean;
  editVectorStore: boolean;
}

const VectorStoreInfoView: React.FC<VectorStoreInfoViewProps> = ({
  vectorStoreId,
  onClose,
  accessToken,
  is_admin,
  editVectorStore,
}) => {
  const [form] = Form.useForm();
  const [vectorStoreDetails, setVectorStoreDetails] = useState<VectorStore | null>(null);
  const [isEditing, setIsEditing] = useState<boolean>(editVectorStore);
  const [metadataString, setMetadataString] = useState<string>("{}");
  const [credentials, setCredentials] = useState<CredentialItem[]>([]);
  const [activeTab, setActiveTab] = useState<string>(editVectorStore ? "details" : "details");
  const [selectedProvider, setSelectedProvider] = useState<string>("bedrock");
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [isDetectingEmbeddingDimension, setIsDetectingEmbeddingDimension] = useState(false);
  const [detectedEmbeddingDimension, setDetectedEmbeddingDimension] = useState<number | null>(null);

  const detectEmbeddingDimension = async (embeddingModel: string | null) => {
    if (!accessToken || !embeddingModel) {
      setDetectedEmbeddingDimension(null);
      return;
    }
    setIsDetectingEmbeddingDimension(true);
    try {
      const dimension = await detectEmbeddingDimensionCall(accessToken, embeddingModel);
      setDetectedEmbeddingDimension(dimension);
    } catch (error) {
      console.error("Error detecting embedding dimension:", error);
      NotificationsManager.fromBackend(`Failed to detect embedding dimension: ${error}`);
      setDetectedEmbeddingDimension(null);
    } finally {
      setIsDetectingEmbeddingDimension(false);
    }
  };

  const fetchVectorStoreDetails = async () => {
    if (!accessToken) return;
    try {
      const response = await vectorStoreInfoCall(accessToken, vectorStoreId);
      if (response && response.vector_store) {
        setVectorStoreDetails(response.vector_store);
        const provider = response.vector_store.custom_llm_provider || "bedrock";
        setSelectedProvider(provider);

        // If metadata exists and is an object, stringify it for display/editing
        if (response.vector_store.vector_store_metadata) {
          const metadata =
            typeof response.vector_store.vector_store_metadata === "string"
              ? JSON.parse(response.vector_store.vector_store_metadata)
              : response.vector_store.vector_store_metadata;
          setMetadataString(JSON.stringify(metadata, null, 2));
        }

        let parsedLitellmParams: Record<string, any> = {};
        if (response.vector_store.litellm_params) {
          if (typeof response.vector_store.litellm_params === "string") {
            try {
              parsedLitellmParams = JSON.parse(response.vector_store.litellm_params);
            } catch (error) {
              console.error("Error parsing litellm_params:", error);
            }
          } else if (typeof response.vector_store.litellm_params === "object") {
            parsedLitellmParams = response.vector_store.litellm_params;
          }
        }

        const embeddingModel =
          parsedLitellmParams.litellm_embedding_model || parsedLitellmParams.embedding_model || null;

        if (editVectorStore) {
          form.setFieldsValue({
            vector_store_id: response.vector_store.vector_store_id,
            custom_llm_provider: response.vector_store.custom_llm_provider,
            vector_store_name: response.vector_store.vector_store_name,
            vector_store_description: response.vector_store.vector_store_description,
            litellm_credential_name: response.vector_store.litellm_credential_name,
            ...parsedLitellmParams,
          });
        }

        if (embeddingModel) {
          detectEmbeddingDimension(embeddingModel);
        }
      }
    } catch (error) {
      console.error("Error fetching vector store details:", error);
      NotificationsManager.fromBackend("Error fetching vector store details: " + error);
    }
  };

  const fetchCredentials = async () => {
    if (!accessToken) return;
    try {
      const response = await credentialListCall(accessToken);
      console.log("List credentials response:", response);
      setCredentials(response.credentials || []);
    } catch (error) {
      console.error("Error fetching credentials:", error);
    }
  };

  useEffect(() => {
    fetchVectorStoreDetails();
    fetchCredentials();
  }, [vectorStoreId, accessToken]);

  useEffect(() => {
    if (!accessToken) return;

    const loadModels = async () => {
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        if (uniqueModels.length > 0) {
          setModelInfo(uniqueModels);
        }
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };

    loadModels();
  }, [accessToken]);

  const handleSave = async (values: any) => {
    if (!accessToken) return;
    try {
      // Parse the metadata JSON string
      let metadata = {};
      try {
        metadata = metadataString ? JSON.parse(metadataString) : {};
      } catch (e) {
        NotificationsManager.fromBackend("Invalid JSON in metadata field");
        return;
      }

      const providerFields = getProviderSpecificFields(values.custom_llm_provider);
      const litellmParams = providerFields.reduce(
        (acc, field) => {
          acc[field.name] = values[field.name];
          return acc;
        },
        {} as Record<string, any>,
      );

      const updateData = {
        vector_store_id: values.vector_store_id,
        custom_llm_provider: values.custom_llm_provider,
        vector_store_name: values.vector_store_name,
        vector_store_description: values.vector_store_description,
        vector_store_metadata: metadata,
        litellm_credential_name: values.litellm_credential_name,
        litellm_params: litellmParams,
      };

      await vectorStoreUpdateCall(accessToken, updateData);
      NotificationsManager.success("Vector store updated successfully");
      setIsEditing(false);
      fetchVectorStoreDetails();
    } catch (error) {
      console.error("Error updating vector store:", error);
      NotificationsManager.fromBackend("Error updating vector store: " + error);
    }
  };

  if (!vectorStoreDetails) {
    return <div>Loading...</div>;
  }

  return (
    <div className="p-4 max-w-full">
      <div className="flex justify-between items-center mb-6">
        <div>
          <Button icon={ArrowLeftIcon} variant="light" className="mb-4" onClick={onClose}>
            Back to Vector Stores
          </Button>
          <Title>Vector Store ID: {vectorStoreDetails.vector_store_id}</Title>
          <Text className="text-gray-500">{vectorStoreDetails.vector_store_description || "No description"}</Text>
        </div>
        {is_admin && !isEditing && <Button onClick={() => setIsEditing(true)}>Edit Vector Store</Button>}
      </div>

      <TabGroup>
        <TabList className="mb-6">
          <Tab>Details</Tab>
          <Tab>Test Vector Store</Tab>
        </TabList>

        <TabPanels>
          {/* Details Tab */}
          <TabPanel>
            {isEditing ? (
              <div>
                <div className="flex justify-between items-center mb-4">
                  <Title>Edit Vector Store</Title>
                </div>
                <Card>
                  <Form form={form} onFinish={handleSave} layout="vertical" initialValues={vectorStoreDetails}>
                    <Form.Item
                      label="Vector Store ID"
                      name="vector_store_id"
                      rules={[{ required: true, message: "Please input a vector store ID" }]}
                    >
                      <Input disabled />
                    </Form.Item>

                    <Form.Item label="Vector Store Name" name="vector_store_name">
                      <Input />
                    </Form.Item>

                    <Form.Item label="Description" name="vector_store_description">
                      <Input.TextArea rows={4} />
                    </Form.Item>

                    <Form.Item
                      label={
                        <span>
                          Provider{" "}
                          <Tooltip title="Select the provider for this vector store">
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </Tooltip>
                        </span>
                      }
                      name="custom_llm_provider"
                      rules={[{ required: true, message: "Please select a provider" }]}
                    >
                      <Select2 onChange={(value) => setSelectedProvider(value)}>
                        {Object.entries(VectorStoreProviders).map(([providerEnum, providerDisplayName]) => {
                          return (
                            <Select2.Option key={providerEnum} value={vectorStoreProviderMap[providerEnum]}>
                              <div className="flex items-center space-x-2">
                                <img
                                  src={vectorStoreProviderLogoMap[providerDisplayName]}
                                  alt={`${providerEnum} logo`}
                                  className="w-5 h-5"
                                  onError={(e) => {
                                    const target = e.target as HTMLImageElement;
                                    const parent = target.parentElement;
                                    if (parent) {
                                      const fallbackDiv = document.createElement("div");
                                      fallbackDiv.className =
                                        "w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs";
                                      fallbackDiv.textContent = providerDisplayName.charAt(0);
                                      parent.replaceChild(fallbackDiv, target);
                                    }
                                  }}
                                />
                                <span>{providerDisplayName}</span>
                              </div>
                            </Select2.Option>
                          );
                        })}
                      </Select2>
                    </Form.Item>

                    {getProviderSpecificFields(selectedProvider).map((field: VectorStoreFieldConfig) => {
                      if (field.type === "select") {
                        const embeddingModels = modelInfo
                          .filter((option: ModelGroup) => option.mode === "embedding")
                          .map((option: ModelGroup) => ({
                            value: option.model_group,
                            label: option.model_group,
                          }));

                        return (
                          <Form.Item
                            key={field.name}
                            label={
                              <span>
                                {field.label}{" "}
                                <Tooltip title={field.tooltip}>
                                  <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                                </Tooltip>
                              </span>
                            }
                            name={field.name}
                            rules={
                              field.required
                                ? [{ required: true, message: `Please select the ${field.label.toLowerCase()}` }]
                                : []
                            }
                          >
                            <Select2
                              placeholder={field.placeholder}
                              showSearch={true}
                              filterOption={(input, option) =>
                                (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
                              }
                              options={embeddingModels}
                              onChange={(value) => {
                                if (field.name === "litellm_embedding_model") {
                                  detectEmbeddingDimension(value);
                                }
                              }}
                            />
                          </Form.Item>
                        );
                      }

                      return (
                        <Form.Item
                          key={field.name}
                          label={
                            <span>
                              {field.label}{" "}
                              <Tooltip title={field.tooltip}>
                                <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                              </Tooltip>
                            </span>
                          }
                          name={field.name}
                          rules={
                            field.required
                              ? [{ required: true, message: `Please input the ${field.label.toLowerCase()}` }]
                              : []
                          }
                        >
                          <Input type={field.type || "text"} placeholder={field.placeholder} />
                        </Form.Item>
                      );
                    })}

                    {detectedEmbeddingDimension !== null && (
                      <Form.Item label="Detected Embedding Dimension">
                        <Input
                          value={detectedEmbeddingDimension}
                          disabled
                          placeholder={isDetectingEmbeddingDimension ? "Detecting..." : ""}
                        />
                      </Form.Item>
                    )}

                    {/* Credentials */}
                    <div className="mb-4">
                      <Text className="text-sm text-gray-500 mb-2">
                        Either select existing credentials OR enter provider credentials below
                      </Text>
                    </div>

                    <Form.Item label="Existing Credentials" name="litellm_credential_name">
                      <Select2
                        showSearch
                        placeholder="Select or search for existing credentials"
                        optionFilterProp="children"
                        filterOption={(input, option) =>
                          (option?.label ?? "").toLowerCase().includes(input.toLowerCase())
                        }
                        options={[
                          { value: null, label: "None" },
                          ...credentials.map((credential) => ({
                            value: credential.credential_name,
                            label: credential.credential_name,
                          })),
                        ]}
                        allowClear
                      />
                    </Form.Item>

                    <div className="flex items-center my-4">
                      <div className="flex-grow border-t border-gray-200"></div>
                      <span className="px-4 text-gray-500 text-sm">OR</span>
                      <div className="flex-grow border-t border-gray-200"></div>
                    </div>

                    <Form.Item
                      label={
                        <span>
                          Metadata{" "}
                          <Tooltip title="JSON metadata for the vector store">
                            <InfoCircleOutlined style={{ marginLeft: "4px" }} />
                          </Tooltip>
                        </span>
                      }
                    >
                      <Input.TextArea
                        rows={4}
                        value={metadataString}
                        onChange={(e) => setMetadataString(e.target.value)}
                        placeholder='{"key": "value"}'
                      />
                    </Form.Item>

                    <div className="flex justify-end space-x-2">
                      <AntButton onClick={() => setIsEditing(false)}>Cancel</AntButton>
                      <AntButton type="primary" htmlType="submit">
                        Save Changes
                      </AntButton>
                    </div>
                  </Form>
                </Card>
              </div>
            ) : (
              <div>
                <div className="flex justify-between items-center mb-4">
                  <Title>Vector Store Details</Title>
                  {is_admin && <Button onClick={() => setIsEditing(true)}>Edit Vector Store</Button>}
                </div>
                <Card>
                  <div className="space-y-4">
                    <div>
                      <Text className="font-medium">ID</Text>
                      <Text>{vectorStoreDetails.vector_store_id}</Text>
                    </div>
                    <div>
                      <Text className="font-medium">Name</Text>
                      <Text>{vectorStoreDetails.vector_store_name || "-"}</Text>
                    </div>
                    <div>
                      <Text className="font-medium">Description</Text>
                      <Text>{vectorStoreDetails.vector_store_description || "-"}</Text>
                    </div>
                    <div>
                      <Text className="font-medium">Provider</Text>
                      <div className="flex items-center space-x-2 mt-1">
                        {(() => {
                          const provider = vectorStoreDetails.custom_llm_provider || "bedrock";
                          const { displayName, logo } = (() => {
                            // Find the enum key by matching vectorStoreProviderMap values
                            const enumKey = Object.keys(vectorStoreProviderMap).find(
                              (key) =>
                                vectorStoreProviderMap[key].toLowerCase() === provider.toLowerCase(),
                            );

                            if (!enumKey) {
                              return { displayName: provider, logo: "" };
                            }

                            // Get the display name from VectorStoreProviders enum and logo from map
                            const displayName =
                              VectorStoreProviders[enumKey as keyof typeof VectorStoreProviders];
                            const logo =
                              vectorStoreProviderLogoMap[
                                displayName as keyof typeof vectorStoreProviderLogoMap
                              ];

                            return { displayName, logo };
                          })();

                          return (
                            <>
                              {logo && (
                                <img
                                  src={logo}
                                  alt={`${displayName} logo`}
                                  className="w-5 h-5"
                                  onError={(e) => {
                                    const target = e.target as HTMLImageElement;
                                    const parent = target.parentElement;
                                    if (parent) {
                                      const fallbackDiv = document.createElement("div");
                                      fallbackDiv.className =
                                        "w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center text-xs";
                                      fallbackDiv.textContent = displayName.charAt(0);
                                      parent.replaceChild(fallbackDiv, target);
                                    }
                                  }}
                                />
                              )}
                              <Badge color="orange">{displayName}</Badge>
                            </>
                          );
                        })()}
                      </div>
                    </div>
                    <div>
                      <Text className="font-medium">Metadata</Text>
                      <div className="bg-gray-50 p-3 rounded mt-2 font-mono text-xs overflow-auto max-h-48">
                        <pre>{metadataString}</pre>
                      </div>
                    </div>
                    <div>
                      <Text className="font-medium">Created</Text>
                      <Text>
                        {vectorStoreDetails.created_at ? new Date(vectorStoreDetails.created_at).toLocaleString() : "-"}
                      </Text>
                    </div>
                    <div>
                      <Text className="font-medium">Last Updated</Text>
                      <Text>
                        {vectorStoreDetails.updated_at ? new Date(vectorStoreDetails.updated_at).toLocaleString() : "-"}
                      </Text>
                    </div>
                  </div>
                </Card>
              </div>
            )}
          </TabPanel>

          {/* Test Tab */}
          <TabPanel>
            <VectorStoreTester vectorStoreId={vectorStoreDetails.vector_store_id} accessToken={accessToken || ""} />
          </TabPanel>
        </TabPanels>
      </TabGroup>
    </div>
  );
};

export default VectorStoreInfoView;
