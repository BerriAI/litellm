import React, { useState, useEffect } from "react";
import { Form, Select as AntdSelect } from "antd";
import type { FormInstance } from "antd";
import { Text, TextInput } from "@tremor/react";
import { modelAvailableCall } from "../networking";
import { all_admin_roles } from "@/utils/roles";
import { fetchAvailableModels, ModelGroup } from "../chat_ui/llm_calls/fetch_models";
import RouterConfigBuilder from "../add_model/router_config_builder";

interface EditAutoRouterTabProps {
  form: FormInstance;
  localModelData: any;
  accessToken: string;
  userRole: string;
  isEditing: boolean;
  setIsDirty: (dirty: boolean) => void;
}

const EditAutoRouterTab: React.FC<EditAutoRouterTabProps> = ({
  form,
  localModelData,
  accessToken,
  userRole,
  isEditing,
  setIsDirty,
}) => {
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [showCustomDefaultModel, setShowCustomDefaultModel] = useState<boolean>(false);
  const [showCustomEmbeddingModel, setShowCustomEmbeddingModel] = useState<boolean>(false);
  const [routerConfig, setRouterConfig] = useState<any>(null);

  const isAdmin = all_admin_roles.includes(userRole);

  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      const response = await modelAvailableCall(accessToken, "", "", false, null, true, true);
      setModelAccessGroups(response["data"].map((model: any) => model["id"]));
    };
    fetchModelAccessGroups();
  }, [accessToken]);

  useEffect(() => {
    const loadModels = async () => {
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        setModelInfo(uniqueModels);
      } catch (error) {
        console.error("Error fetching model info for auto router edit:", error);
      }
    };
    loadModels();
  }, [accessToken]);

  // Initialize router config when localModelData changes
  useEffect(() => {
    if (localModelData?.litellm_params?.auto_router_config) {
      try {
        const autoRouterConfig = localModelData.litellm_params.auto_router_config;
        const parsedConfig = typeof autoRouterConfig === 'string' 
          ? JSON.parse(autoRouterConfig) 
          : autoRouterConfig;
        setRouterConfig(parsedConfig);
      } catch (error) {
        console.error("Error parsing auto router config:", error);
      }
    }
  }, [localModelData]);

  const handleRouterConfigChange = (config: any) => {
    setRouterConfig(config);
    form.setFieldValue('auto_router_config', config);
    setIsDirty(true);
  };

  return (
    <div className="space-y-6">
      <div>
        <Text className="text-lg font-semibold">Auto Router Configuration</Text>
        <Text className="text-gray-600">
          Configure the auto routing logic that automatically selects the best model based on user input patterns and semantic matching.
        </Text>
      </div>

      <div className="space-y-4">
        {/* Auto Router Name */}
        <div>
          <Text className="font-medium">Auto Router Name</Text>
          {isEditing ? (
            <Form.Item name="model_name" className="mb-0">
              <TextInput 
                placeholder="e.g., auto_router_1, smart_routing"
                onChange={() => setIsDirty(true)}
              />
            </Form.Item>
          ) : (
            <div className="mt-1 p-2 bg-gray-50 rounded">
              {localModelData?.model_name || "Not Set"}
            </div>
          )}
        </div>

        {/* Router Configuration Builder */}
        {isEditing ? (
          <div className="w-full">
            <RouterConfigBuilder
              modelInfo={modelInfo}
              value={routerConfig}
              onChange={handleRouterConfigChange}
            />
          </div>
        ) : (
          <div>
            <Text className="font-medium">Router Configuration</Text>
            <div className="mt-1 p-2 bg-gray-50 rounded">
              {routerConfig ? (
                <div className="space-y-2">
                  <Text className="text-sm font-medium">Routes: {routerConfig.routes?.length || 0}</Text>
                  {routerConfig.routes?.map((route: any, index: number) => (
                    <div key={index} className="text-sm bg-white p-2 rounded border">
                      <div><strong>Model:</strong> {route.name}</div>
                      <div><strong>Description:</strong> {route.description}</div>
                      <div><strong>Utterances:</strong> {route.utterances?.length || 0} examples</div>
                      <div><strong>Score Threshold:</strong> {route.score_threshold}</div>
                    </div>
                  ))}
                </div>
              ) : (
                "No router configuration"
              )}
            </div>
          </div>
        )}

        {/* Default Model */}
        <div>
          <Text className="font-medium">Default Model</Text>
          {isEditing ? (
            <>
              <Form.Item name="auto_router_default_model" className="mb-0">
                <AntdSelect
                  placeholder="Select a default model"
                  onChange={(value) => {
                    setShowCustomDefaultModel(value === 'custom');
                    setIsDirty(true);
                  }}
                  options={[
                    ...Array.from(new Set(modelInfo.map(option => option.model_group)))
                      .map((model_group) => ({
                        value: model_group,
                        label: model_group,
                      })),
                    { value: 'custom', label: 'Enter custom model name' }
                  ]}
                  style={{ width: "100%" }}
                  showSearch={true}
                />
              </Form.Item>
              {showCustomDefaultModel && (
                <Form.Item name="custom_default_model" className="mb-0 mt-2">
                  <TextInput 
                    placeholder="Enter custom model name"
                    onChange={(e) => {
                      form.setFieldValue('auto_router_default_model', e.target.value);
                      setIsDirty(true);
                    }}
                  />
                </Form.Item>
              )}
            </>
          ) : (
            <div className="mt-1 p-2 bg-gray-50 rounded">
              {localModelData?.litellm_params?.auto_router_default_model || "Not Set"}
            </div>
          )}
        </div>

        {/* Embedding Model */}
        <div>
          <Text className="font-medium">Embedding Model</Text>
          {isEditing ? (
            <>
              <Form.Item name="auto_router_embedding_model" className="mb-0">
                <AntdSelect
                  placeholder="Select an embedding model (optional)"
                  onChange={(value) => {
                    setShowCustomEmbeddingModel(value === 'custom');
                    setIsDirty(true);
                  }}
                  options={[
                    ...Array.from(new Set(modelInfo.map(option => option.model_group)))
                      .map((model_group) => ({
                        value: model_group,
                        label: model_group,
                      })),
                    { value: 'custom', label: 'Enter custom model name' }
                  ]}
                  style={{ width: "100%" }}
                  showSearch={true}
                  allowClear
                />
              </Form.Item>
              {showCustomEmbeddingModel && (
                <Form.Item name="custom_embedding_model" className="mb-0 mt-2">
                  <TextInput 
                    placeholder="Enter custom embedding model name"
                    onChange={(e) => {
                      form.setFieldValue('auto_router_embedding_model', e.target.value);
                      setIsDirty(true);
                    }}
                  />
                </Form.Item>
              )}
            </>
          ) : (
            <div className="mt-1 p-2 bg-gray-50 rounded">
              {localModelData?.litellm_params?.auto_router_embedding_model || "Not Set"}
            </div>
          )}
        </div>

        {/* Model Access Groups - Admin only */}
        {isAdmin && (
          <div>
            <Text className="font-medium">Model Access Groups</Text>
            {isEditing ? (
              <Form.Item name="model_access_group" className="mb-0">
                <AntdSelect
                  mode="tags"
                  showSearch
                  placeholder="Select existing groups or type to create new ones"
                  optionFilterProp="children"
                  tokenSeparators={[',']}
                  options={modelAccessGroups.map((group) => ({
                    value: group,
                    label: group
                  }))}
                  maxTagCount="responsive"
                  allowClear
                  onChange={() => setIsDirty(true)}
                />
              </Form.Item>
            ) : (
              <div className="mt-1 p-2 bg-gray-50 rounded">
                {localModelData?.model_info?.access_groups ? (
                  Array.isArray(localModelData.model_info.access_groups) ? (
                    localModelData.model_info.access_groups.length > 0 ? (
                      <div className="flex flex-wrap gap-1">
                        {localModelData.model_info.access_groups.map(
                          (group: string, index: number) => (
                            <span
                              key={index}
                              className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800"
                            >
                              {group}
                            </span>
                          )
                        )}
                      </div>
                    ) : (
                      "No groups assigned"
                    )
                  ) : (
                    localModelData.model_info.access_groups
                  )
                ) : (
                  "Not Set"
                )}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default EditAutoRouterTab; 