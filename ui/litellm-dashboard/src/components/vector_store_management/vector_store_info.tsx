import React, { useState, useEffect } from "react";
import { Card, Text, Title, Button, Badge, TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import { Form, Input, Select as Select2, Tooltip, Button as AntButton } from "antd";
import { InfoCircleOutlined } from "@ant-design/icons";
import { ArrowLeftIcon } from "@heroicons/react/outline";
import { vectorStoreInfoCall, vectorStoreUpdateCall, credentialListCall, CredentialItem } from "../networking";
import { VectorStore } from "./types";
import { Providers, providerLogoMap, provider_map } from "../provider_info_helpers";
import VectorStoreTester from "./VectorStoreTester";
import NotificationsManager from "../molecules/notifications_manager";

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

  const fetchVectorStoreDetails = async () => {
    if (!accessToken) return;
    try {
      const response = await vectorStoreInfoCall(accessToken, vectorStoreId);
      if (response && response.vector_store) {
        setVectorStoreDetails(response.vector_store);

        // If metadata exists and is an object, stringify it for display/editing
        if (response.vector_store.vector_store_metadata) {
          const metadata =
            typeof response.vector_store.vector_store_metadata === "string"
              ? JSON.parse(response.vector_store.vector_store_metadata)
              : response.vector_store.vector_store_metadata;
          setMetadataString(JSON.stringify(metadata, null, 2));
        }

        if (editVectorStore) {
          form.setFieldsValue({
            vector_store_id: response.vector_store.vector_store_id,
            custom_llm_provider: response.vector_store.custom_llm_provider,
            vector_store_name: response.vector_store.vector_store_name,
            vector_store_description: response.vector_store.vector_store_description,
          });
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

      const updateData = {
        vector_store_id: values.vector_store_id,
        custom_llm_provider: values.custom_llm_provider,
        vector_store_name: values.vector_store_name,
        vector_store_description: values.vector_store_description,
        vector_store_metadata: metadata,
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
                      <Select2>
                        {Object.entries(Providers).map(([providerEnum, providerDisplayName]) => {
                          // Currently only showing Bedrock since it's the only supported provider
                          if (providerEnum === "Bedrock") {
                            return (
                              <Select2.Option key={providerEnum} value={provider_map[providerEnum]}>
                                <div className="flex items-center space-x-2">
                                  <img
                                    src={providerLogoMap[providerDisplayName]}
                                    alt={`${providerEnum} logo`}
                                    className="w-5 h-5"
                                    onError={(e) => {
                                      // Create a div with provider initial as fallback
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
                          }
                          return null;
                        })}
                      </Select2>
                    </Form.Item>

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
                            // Find the enum key by matching provider_map values
                            const enumKey = Object.keys(provider_map).find(
                              (key) => provider_map[key].toLowerCase() === provider.toLowerCase(),
                            );

                            if (!enumKey) {
                              return { displayName: provider, logo: "" };
                            }

                            // Get the display name from Providers enum and logo from map
                            const displayName = Providers[enumKey as keyof typeof Providers];
                            const logo = providerLogoMap[displayName];

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
                              <Badge color="blue">{displayName}</Badge>
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
