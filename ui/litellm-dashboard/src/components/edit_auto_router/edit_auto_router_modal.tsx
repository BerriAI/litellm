import React, { useEffect, useState } from "react";
import { Modal, Form, Button, Select as AntdSelect } from "antd";
import { Text, TextInput } from "@tremor/react";
import { modelAvailableCall, modelPatchUpdateCall } from "../networking";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import RouterConfigBuilder from "../add_model/RouterConfigBuilder";
import ComplexityRouterConfig, {
  ComplexityRouterConfigValue,
  DEFAULT_ADAPTIVE_WEIGHTS,
  DEFAULT_TIER_DISTANCE_PENALTY,
} from "../add_model/ComplexityRouterConfig";
import NotificationsManager from "../molecules/notifications_manager";

const isComplexityRouterModel = (modelData: any): boolean =>
  modelData?.litellm_params?.model?.startsWith("auto_router/complexity_router") ||
  modelData?.litellm_params?.complexity_router_config != null;

const normalizeTierModels = (value: unknown): string[] => {
  if (Array.isArray(value)) return value;
  if (typeof value === "string" && value) return [value];
  return [];
};

interface EditAutoRouterModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: (updatedModel: any) => void;
  modelData: any;
  accessToken: string;
  userRole: string;
}

const MANAGED_COMPLEXITY_ROUTER_KEYS = new Set([
  "tiers",
  "classifier_type",
  "classifier_llm_config",
  "adaptive",
  "adaptive_weights",
  "tier_distance_penalty",
  "adaptive_eligible",
]);

const toRecord = (value: unknown): Record<string, unknown> => {
  const parsed: unknown = typeof value === "string" ? JSON.parse(value) : value;
  return typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)
    ? (parsed as Record<string, unknown>)
    : {};
};

export const buildUpdatedComplexityRouterConfig = (
  storedConfig: unknown,
  value: ComplexityRouterConfigValue,
  customTechnicalKeywords?: string[],
): Record<string, unknown> => {
  const preservedConfig = Object.fromEntries(
    Object.entries(toRecord(storedConfig)).filter(
      ([key]) =>
        !MANAGED_COMPLEXITY_ROUTER_KEYS.has(key) &&
        (customTechnicalKeywords === undefined || key !== "custom_technical_keywords"),
    ),
  );
  const adaptiveEligible = value.adaptive_eligible ?? "all";

  return {
    ...preservedConfig,
    tiers: value.tiers,
    classifier_type: value.classifier_type,
    ...(value.classifier_type === "llm" ? { classifier_llm_config: value.classifier_llm_config } : {}),
    ...(customTechnicalKeywords &&
      customTechnicalKeywords.length > 0 && {
        custom_technical_keywords: customTechnicalKeywords,
      }),
    ...(value.adaptive && {
      adaptive: true,
      adaptive_weights: value.adaptive_weights ?? DEFAULT_ADAPTIVE_WEIGHTS,
      ...(adaptiveEligible === "all" && {
        tier_distance_penalty: value.tier_distance_penalty ?? DEFAULT_TIER_DISTANCE_PENALTY,
      }),
      adaptive_eligible: adaptiveEligible,
    }),
  };
};

const EditAutoRouterModal: React.FC<EditAutoRouterModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
  modelData,
  accessToken,
  userRole,
}) => {
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [showCustomDefaultModel, setShowCustomDefaultModel] = useState<boolean>(false);
  const [showCustomEmbeddingModel, setShowCustomEmbeddingModel] = useState<boolean>(false);
  const [routerConfig, setRouterConfig] = useState<any>(null);
  const [customTechnicalKeywords, setCustomTechnicalKeywords] = useState<string[]>([]);
  const [complexityRouterConfig, setComplexityRouterConfig] = useState<ComplexityRouterConfigValue>({
    tiers: { SIMPLE: [], MEDIUM: [], COMPLEX: [], REASONING: [] },
    classifier_type: "heuristic",
  });
  const isComplexityRouter = isComplexityRouterModel(modelData);

  useEffect(() => {
    if (isVisible && modelData) {
      initializeForm();
    }
  }, [isVisible, modelData]);

  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      if (!accessToken) return;
      try {
        const response = await modelAvailableCall(accessToken, "", "", false, null, true, true);
        setModelAccessGroups(response["data"].map((model: any) => model["id"]));
      } catch (error) {
        console.error("Error fetching model access groups:", error);
      }
    };

    const loadModels = async () => {
      if (!accessToken) return;
      try {
        const uniqueModels = await fetchAvailableModels(accessToken);
        setModelInfo(uniqueModels);
      } catch (error) {
        console.error("Error fetching model info:", error);
      }
    };

    if (isVisible) {
      fetchModelAccessGroups();
      loadModels();
    }
  }, [isVisible, accessToken]);

  const initializeForm = () => {
    try {
      if (isComplexityRouterModel(modelData)) {
        // Parse the complexity_router_config if it exists and is a string
        let parsedConfig = modelData.litellm_params?.complexity_router_config || {};
        if (typeof parsedConfig === "string") {
          parsedConfig = JSON.parse(parsedConfig);
        }

        setComplexityRouterConfig({
          tiers: {
            SIMPLE: normalizeTierModels(parsedConfig.tiers?.SIMPLE),
            MEDIUM: normalizeTierModels(parsedConfig.tiers?.MEDIUM),
            COMPLEX: normalizeTierModels(parsedConfig.tiers?.COMPLEX),
            REASONING: normalizeTierModels(parsedConfig.tiers?.REASONING),
          },
          classifier_type: parsedConfig.classifier_type || "heuristic",
          classifier_llm_config: parsedConfig.classifier_llm_config,
          adaptive: parsedConfig.adaptive || false,
          adaptive_weights: parsedConfig.adaptive_weights,
          tier_distance_penalty: parsedConfig.tier_distance_penalty,
          adaptive_eligible: parsedConfig.adaptive_eligible || "all",
        });
        setCustomTechnicalKeywords(
          Array.isArray(parsedConfig.custom_technical_keywords) ? parsedConfig.custom_technical_keywords : [],
        );

        form.setFieldsValue({
          auto_router_name: modelData.model_name,
          model_access_group: modelData.model_info?.access_groups || [],
        });
        return;
      }

      // Parse the auto_router_config if it exists and is a string
      let parsedConfig = null;
      if (modelData.litellm_params?.auto_router_config) {
        if (typeof modelData.litellm_params.auto_router_config === "string") {
          parsedConfig = JSON.parse(modelData.litellm_params.auto_router_config);
        } else {
          parsedConfig = modelData.litellm_params.auto_router_config;
        }
      }

      setRouterConfig(parsedConfig);

      // Set form values
      form.setFieldsValue({
        auto_router_name: modelData.model_name,
        auto_router_default_model: modelData.litellm_params?.auto_router_default_model || "",
        auto_router_embedding_model: modelData.litellm_params?.auto_router_embedding_model || "",
        model_access_group: modelData.model_info?.access_groups || [],
      });

      // Check if using custom models
      const allModelGroups = new Set(modelInfo.map((model) => model.model_group));
      setShowCustomDefaultModel(!allModelGroups.has(modelData.litellm_params?.auto_router_default_model));
      setShowCustomEmbeddingModel(!allModelGroups.has(modelData.litellm_params?.auto_router_embedding_model));
    } catch (error) {
      console.error("Error parsing auto router config:", error);
      NotificationsManager.fromBackend("Error loading auto router configuration");
    }
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      const values = await form.validateFields();

      if (isComplexityRouter) {
        const { tiers, classifier_type, classifier_llm_config } = complexityRouterConfig;
        if (Object.values(tiers).every((models) => models.length === 0)) {
          NotificationsManager.fromBackend("Please select at least one model for a complexity tier");
          return;
        }
        if (classifier_type === "llm" && !classifier_llm_config?.model) {
          NotificationsManager.fromBackend("Please select a classifier model, or switch back to Heuristic");
          return;
        }

        const defaultModel = tiers.MEDIUM[0] || tiers.SIMPLE[0] || tiers.COMPLEX[0] || tiers.REASONING[0];
        const updatedLitellmParams = {
          ...modelData.litellm_params,
          complexity_router_config: buildUpdatedComplexityRouterConfig(
            modelData.litellm_params?.complexity_router_config,
            complexityRouterConfig,
            customTechnicalKeywords,
          ),
          complexity_router_default_model: defaultModel,
        };
        const updatedModelInfo = {
          ...modelData.model_info,
          access_groups: values.model_access_group || [],
        };

        await modelPatchUpdateCall(
          accessToken,
          { model_name: values.auto_router_name, litellm_params: updatedLitellmParams, model_info: updatedModelInfo },
          modelData.model_info.id,
        );

        NotificationsManager.success("Auto router configuration updated successfully");
        onSuccess({
          ...modelData,
          model_name: values.auto_router_name,
          litellm_params: updatedLitellmParams,
          model_info: updatedModelInfo,
        });
        onCancel();
        return;
      }

      // Prepare the updated litellm_params
      const updatedLitellmParams = {
        ...modelData.litellm_params,
        auto_router_config: JSON.stringify(routerConfig),
        auto_router_default_model: values.auto_router_default_model,
        auto_router_embedding_model: values.auto_router_embedding_model || undefined,
      };

      // Prepare updated model_info
      const updatedModelInfo = {
        ...modelData.model_info,
        access_groups: values.model_access_group || [],
      };

      const updateData = {
        model_name: values.auto_router_name,
        litellm_params: updatedLitellmParams,
        model_info: updatedModelInfo,
      };

      await modelPatchUpdateCall(accessToken, updateData, modelData.model_info.id);

      const updatedModelData = {
        ...modelData,
        model_name: values.auto_router_name,
        litellm_params: updatedLitellmParams,
        model_info: updatedModelInfo,
      };

      NotificationsManager.success("Auto router configuration updated successfully");
      onSuccess(updatedModelData);
      onCancel();
    } catch (error) {
      console.error("Error updating auto router:", error);
      NotificationsManager.fromBackend("Failed to update auto router configuration");
    } finally {
      setLoading(false);
    }
  };

  const modelOptions = modelInfo.map((model) => ({
    value: model.model_group,
    label: model.model_group,
  }));

  return (
    <Modal
      title="Edit Auto Router Configuration"
      open={isVisible}
      onCancel={onCancel}
      footer={[
        <Button key="cancel" onClick={onCancel}>
          Cancel
        </Button>,
        <Button key="submit" loading={loading} onClick={handleSubmit}>
          Save Changes
        </Button>,
      ]}
      width={1000}
      destroyOnHidden
    >
      <div className="space-y-6">
        <Text className="text-gray-600">
          Edit the auto router configuration including routing logic, default models, and access settings.
        </Text>

        <Form form={form} layout="vertical" className="space-y-4">
          {/* Auto Router Name */}
          <Form.Item
            label="Auto Router Name"
            name="auto_router_name"
            rules={[{ required: true, message: "Auto router name is required" }]}
          >
            <TextInput placeholder="e.g., auto_router_1, smart_routing" />
          </Form.Item>

          {isComplexityRouter ? (
            /* Complexity Router Configuration */
            <div className="w-full">
              <ComplexityRouterConfig
                modelInfo={modelInfo}
                value={complexityRouterConfig}
                onChange={(config) => {
                  setComplexityRouterConfig(config);
                }}
                customTechnicalKeywords={customTechnicalKeywords}
                onCustomTechnicalKeywordsChange={setCustomTechnicalKeywords}
              />
            </div>
          ) : (
            <>
              {/* Router Configuration Builder */}
              <div className="w-full">
                <RouterConfigBuilder
                  modelInfo={modelInfo}
                  value={routerConfig}
                  onChange={(config) => {
                    setRouterConfig(config);
                  }}
                />
              </div>

              {/* Default Model */}
              <Form.Item
                label="Default Model"
                name="auto_router_default_model"
                rules={[{ required: true, message: "Default model is required" }]}
              >
                <AntdSelect
                  placeholder="Select a default model"
                  onChange={(value) => {
                    setShowCustomDefaultModel(value === "custom");
                  }}
                  options={[...modelOptions, { value: "custom", label: "Enter custom model name" }]}
                  showSearch={true}
                />
              </Form.Item>

              {/* Embedding Model */}
              <Form.Item
                label="Embedding Model"
                name="auto_router_embedding_model"
                rules={[{ required: true, message: "Embedding model is required" }]}
              >
                <AntdSelect
                  placeholder="Select an embedding model"
                  onChange={(value) => {
                    setShowCustomEmbeddingModel(value === "custom");
                  }}
                  options={[...modelOptions, { value: "custom", label: "Enter custom model name" }]}
                  showSearch={true}
                />
              </Form.Item>
            </>
          )}

          {/* Model Access Groups - Admin only */}
          {userRole === "Admin" && (
            <Form.Item
              label="Model Access Groups"
              name="model_access_group"
              tooltip="Control who can access this auto router"
            >
              <AntdSelect
                mode="tags"
                showSearch
                placeholder="Select existing groups or type to create new ones"
                optionFilterProp="children"
                tokenSeparators={[","]}
                options={modelAccessGroups.map((group) => ({
                  value: group,
                  label: group,
                }))}
                maxTagCount="responsive"
                allowClear
              />
            </Form.Item>
          )}
        </Form>
      </div>
    </Modal>
  );
};

export default EditAutoRouterModal;
