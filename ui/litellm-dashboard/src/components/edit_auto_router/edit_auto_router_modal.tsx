import React, { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import { Modal, Form, Button, Select as AntdSelect } from "antd";
import { Text, TextInput } from "@tremor/react";
import { modelAvailableCall, modelPatchUpdateCall } from "../networking";
import { fetchAvailableModels, ModelGroup } from "../playground/llm_calls/fetch_models";
import RouterConfigBuilder from "../add_model/RouterConfigBuilder";
import NotificationsManager from "../molecules/notifications_manager";

interface EditAutoRouterModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: (updatedModel: any) => void;
  modelData: any;
  accessToken: string;
  userRole: string;
}

const EditAutoRouterModal: React.FC<EditAutoRouterModalProps> = ({
  isVisible,
  onCancel,
  onSuccess,
  modelData,
  accessToken,
  userRole,
}) => {
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [showCustomDefaultModel, setShowCustomDefaultModel] = useState<boolean>(false);
  const [showCustomEmbeddingModel, setShowCustomEmbeddingModel] = useState<boolean>(false);
  const [routerConfig, setRouterConfig] = useState<any>(null);

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
      NotificationsManager.fromBackend(t("editAutoRouter.editAutoRouterModal.errorLoadingConfig"));
    }
  };

  const handleSubmit = async () => {
    try {
      setLoading(true);
      const values = await form.validateFields();

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

      NotificationsManager.success(t("editAutoRouter.editAutoRouterModal.updateSuccess"));
      onSuccess(updatedModelData);
      onCancel();
    } catch (error) {
      console.error("Error updating auto router:", error);
      NotificationsManager.fromBackend(t("editAutoRouter.editAutoRouterModal.updateFailed"));
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
      title={t("editAutoRouter.editAutoRouterModal.title")}
      open={isVisible}
      onCancel={onCancel}
      footer={[
        <Button key="cancel" onClick={onCancel}>
          {t("common.cancel")}
        </Button>,
        <Button key="submit" loading={loading} onClick={handleSubmit}>
          {t("editAutoRouter.editAutoRouterModal.saveChanges")}
        </Button>,
      ]}
      width={1000}
      destroyOnHidden
    >
      <div className="space-y-6">
        <Text className="text-gray-600">{t("editAutoRouter.editAutoRouterModal.description")}</Text>

        <Form form={form} layout="vertical" className="space-y-4">
          {/* Auto Router Name */}
          <Form.Item
            label={t("editAutoRouter.editAutoRouterModal.fieldAutoRouterName")}
            name="auto_router_name"
            rules={[{ required: true, message: t("editAutoRouter.editAutoRouterModal.ruleAutoRouterName") }]}
          >
            <TextInput placeholder={t("editAutoRouter.editAutoRouterModal.placeholderAutoRouterName")} />
          </Form.Item>

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
            label={t("editAutoRouter.editAutoRouterModal.fieldDefaultModel")}
            name="auto_router_default_model"
            rules={[{ required: true, message: t("editAutoRouter.editAutoRouterModal.ruleDefaultModel") }]}
          >
            <AntdSelect
              placeholder={t("editAutoRouter.editAutoRouterModal.placeholderDefaultModel")}
              onChange={(value) => {
                setShowCustomDefaultModel(value === "custom");
              }}
              options={[
                ...modelOptions,
                { value: "custom", label: t("editAutoRouter.editAutoRouterModal.enterCustomModelName") },
              ]}
              showSearch={true}
            />
          </Form.Item>

          {/* Embedding Model */}
          <Form.Item
            label={t("editAutoRouter.editAutoRouterModal.fieldEmbeddingModel")}
            name="auto_router_embedding_model"
          >
            <AntdSelect
              placeholder={t("editAutoRouter.editAutoRouterModal.placeholderEmbeddingModel")}
              onChange={(value) => {
                setShowCustomEmbeddingModel(value === "custom");
              }}
              options={[
                ...modelOptions,
                { value: "custom", label: t("editAutoRouter.editAutoRouterModal.enterCustomModelName") },
              ]}
              showSearch={true}
              allowClear
            />
          </Form.Item>

          {/* Model Access Groups - Admin only */}
          {userRole === "Admin" && (
            <Form.Item
              label={t("editAutoRouter.editAutoRouterModal.fieldModelAccessGroups")}
              name="model_access_group"
              tooltip={t("editAutoRouter.editAutoRouterModal.tooltipModelAccessGroups")}
            >
              <AntdSelect
                mode="tags"
                showSearch
                placeholder={t("editAutoRouter.editAutoRouterModal.placeholderModelAccessGroups")}
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
