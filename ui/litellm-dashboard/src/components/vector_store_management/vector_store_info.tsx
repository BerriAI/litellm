import React, { useCallback, useState, useEffect } from "react";
// eslint-disable-next-line litellm-ui/no-banned-ui-imports
import { TabGroup, TabList, Tab, TabPanels, TabPanel } from "@tremor/react";
import {
  Form,
  Input as AntInput,
  Select as Select2,
} from "antd";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { ArrowLeft, Info } from "lucide-react";
import {
  vectorStoreInfoCall,
  vectorStoreUpdateCall,
  credentialListCall,
  CredentialItem,
} from "../networking";
import { VectorStore } from "./types";
import {
  Providers,
  providerLogoMap,
  provider_map,
} from "../provider_info_helpers";
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

  const fetchVectorStoreDetails = useCallback(async () => {
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
      NotificationsManager.fromBackend(
        "Error fetching vector store details: " + error,
      );
    }
  }, [accessToken, editVectorStore, form, vectorStoreId]);

  const fetchCredentials = useCallback(async () => {
    if (!accessToken) return;
    try {
      const response = await credentialListCall(accessToken);
      setCredentials(response.credentials || []);
    } catch (error) {
      console.error("Error fetching credentials:", error);
    }
  }, [accessToken]);

  useEffect(() => {
    fetchVectorStoreDetails();
    fetchCredentials();
  }, [fetchVectorStoreDetails, fetchCredentials]);

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleSave = async (values: any) => {
    if (!accessToken) return;
    try {
      // Parse the metadata JSON string
      let metadata = {};
      try {
        metadata = metadataString ? JSON.parse(metadataString) : {};
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
      } catch (_e) {
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
          <Button variant="ghost" className="mb-4" onClick={onClose}>
            <ArrowLeft className="h-4 w-4" />
            Back to Vector Stores
          </Button>
          <h1 className="text-2xl font-semibold">
            Vector Store ID: {vectorStoreDetails.vector_store_id}
          </h1>
          <p className="text-muted-foreground">
            {vectorStoreDetails.vector_store_description || "No description"}
          </p>
        </div>
        {is_admin && !isEditing && (
          <Button onClick={() => setIsEditing(true)}>Edit Vector Store</Button>
        )}
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
                  <h2 className="text-lg font-semibold">Edit Vector Store</h2>
                </div>
                <Card className="p-4">
                  <Form
                    form={form}
                    onFinish={handleSave}
                    layout="vertical"
                    initialValues={vectorStoreDetails}
                  >
                    <Form.Item
                      label="Vector Store ID"
                      name="vector_store_id"
                      rules={[
                        {
                          required: true,
                          message: "Please input a vector store ID",
                        },
                      ]}
                    >
                      <AntInput disabled />
                    </Form.Item>

                    <Form.Item
                      label="Vector Store Name"
                      name="vector_store_name"
                    >
                      <AntInput />
                    </Form.Item>

                    <Form.Item
                      label="Description"
                      name="vector_store_description"
                    >
                      <AntInput.TextArea rows={4} />
                    </Form.Item>

                    <Form.Item
                      label={
                        <span>
                          Provider{" "}
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
                              </TooltipTrigger>
                              <TooltipContent>
                                Select the provider for this vector store
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
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
                              <Select2.Option
                                key={providerEnum}
                                value={provider_map[providerEnum]}
                              >
                                <div className="flex items-center space-x-2">
                                  {/* eslint-disable-next-line @next/next/no-img-element */}
                                  <img
                                    src={providerLogoMap[providerDisplayName]}
                                    alt={`${providerEnum} logo`}
                                    className="w-5 h-5"
                                    onError={(e) => {
                                      const target =
                                        e.target as HTMLImageElement;
                                      const parent = target.parentElement;
                                      if (parent) {
                                        const fallbackDiv =
                                          document.createElement("div");
                                        fallbackDiv.className =
                                          "w-5 h-5 rounded-full bg-muted flex items-center justify-center text-xs";
                                        fallbackDiv.textContent =
                                          providerDisplayName.charAt(0);
                                        parent.replaceChild(
                                          fallbackDiv,
                                          target,
                                        );
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

                    <div className="mb-4">
                      <p className="text-sm text-muted-foreground mb-2">
                        Either select existing credentials OR enter provider
                        credentials below
                      </p>
                    </div>

                    <Form.Item
                      label="Existing Credentials"
                      name="litellm_credential_name"
                    >
                      <Select2
                        showSearch
                        placeholder="Select or search for existing credentials"
                        optionFilterProp="children"
                        filterOption={(input, option) =>
                          (option?.label ?? "")
                            .toLowerCase()
                            .includes(input.toLowerCase())
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
                      <div className="flex-grow border-t border-border"></div>
                      <span className="px-4 text-muted-foreground text-sm">
                        OR
                      </span>
                      <div className="flex-grow border-t border-border"></div>
                    </div>

                    <Form.Item
                      label={
                        <span>
                          Metadata{" "}
                          <TooltipProvider>
                            <Tooltip>
                              <TooltipTrigger asChild>
                                <Info className="ml-1 h-3 w-3 inline text-muted-foreground" />
                              </TooltipTrigger>
                              <TooltipContent>
                                JSON metadata for the vector store
                              </TooltipContent>
                            </Tooltip>
                          </TooltipProvider>
                        </span>
                      }
                    >
                      <AntInput.TextArea
                        rows={4}
                        value={metadataString}
                        onChange={(e) => setMetadataString(e.target.value)}
                        placeholder='{"key": "value"}'
                      />
                    </Form.Item>

                    <div className="flex justify-end space-x-2">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => setIsEditing(false)}
                      >
                        Cancel
                      </Button>
                      <Button type="submit">Save Changes</Button>
                    </div>
                  </Form>
                </Card>
              </div>
            ) : (
              <div>
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-lg font-semibold">
                    Vector Store Details
                  </h2>
                  {is_admin && (
                    <Button onClick={() => setIsEditing(true)}>
                      Edit Vector Store
                    </Button>
                  )}
                </div>
                <Card className="p-4">
                  <div className="space-y-4">
                    <div>
                      <p className="font-medium">ID</p>
                      <p>{vectorStoreDetails.vector_store_id}</p>
                    </div>
                    <div>
                      <p className="font-medium">Name</p>
                      <p>{vectorStoreDetails.vector_store_name || "-"}</p>
                    </div>
                    <div>
                      <p className="font-medium">Description</p>
                      <p>
                        {vectorStoreDetails.vector_store_description || "-"}
                      </p>
                    </div>
                    <div>
                      <p className="font-medium">Provider</p>
                      <div className="flex items-center space-x-2 mt-1">
                        {(() => {
                          const provider =
                            vectorStoreDetails.custom_llm_provider || "bedrock";
                          const { displayName, logo } = (() => {
                            const enumKey = Object.keys(provider_map).find(
                              (key) =>
                                provider_map[key].toLowerCase() ===
                                provider.toLowerCase(),
                            );

                            if (!enumKey) {
                              return { displayName: provider, logo: "" };
                            }

                            const displayName =
                              Providers[enumKey as keyof typeof Providers];
                            const logo = providerLogoMap[displayName];

                            return { displayName, logo };
                          })();

                          return (
                            <>
                              {logo && (
                                // eslint-disable-next-line @next/next/no-img-element
                                <img
                                  src={logo}
                                  alt={`${displayName} logo`}
                                  className="w-5 h-5"
                                  onError={(e) => {
                                    const target =
                                      e.target as HTMLImageElement;
                                    const parent = target.parentElement;
                                    if (parent) {
                                      const fallbackDiv =
                                        document.createElement("div");
                                      fallbackDiv.className =
                                        "w-5 h-5 rounded-full bg-muted flex items-center justify-center text-xs";
                                      fallbackDiv.textContent =
                                        displayName.charAt(0);
                                      parent.replaceChild(fallbackDiv, target);
                                    }
                                  }}
                                />
                              )}
                              <Badge className="bg-blue-100 text-blue-700 dark:bg-blue-950 dark:text-blue-300">
                                {displayName}
                              </Badge>
                            </>
                          );
                        })()}
                      </div>
                    </div>
                    <div>
                      <p className="font-medium">Metadata</p>
                      <div className="bg-muted p-3 rounded mt-2 font-mono text-xs overflow-auto max-h-48">
                        <pre>{metadataString}</pre>
                      </div>
                    </div>
                    <div>
                      <p className="font-medium">Created</p>
                      <p>
                        {vectorStoreDetails.created_at
                          ? new Date(
                              vectorStoreDetails.created_at,
                            ).toLocaleString()
                          : "-"}
                      </p>
                    </div>
                    <div>
                      <p className="font-medium">Last Updated</p>
                      <p>
                        {vectorStoreDetails.updated_at
                          ? new Date(
                              vectorStoreDetails.updated_at,
                            ).toLocaleString()
                          : "-"}
                      </p>
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
