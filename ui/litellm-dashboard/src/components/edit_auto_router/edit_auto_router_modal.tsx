import React, { useEffect, useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Form, Select as AntdSelect } from "antd";
import { modelAvailableCall, modelPatchUpdateCall } from "../networking";
import {
  fetchAvailableModels,
  ModelGroup,
} from "../playground/llm_calls/fetch_models";
import RouterConfigBuilder from "../add_model/RouterConfigBuilder";
import NotificationsManager from "../molecules/notifications_manager";

interface EditAutoRouterModalProps {
  isVisible: boolean;
  onCancel: () => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onSuccess: (updatedModel: any) => void;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
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
  const [form] = Form.useForm();
  const [loading, setLoading] = useState(false);
  const [modelAccessGroups, setModelAccessGroups] = useState<string[]>([]);
  const [modelInfo, setModelInfo] = useState<ModelGroup[]>([]);
  const [_showCustomDefaultModel, setShowCustomDefaultModel] =
    useState<boolean>(false);
  const [_showCustomEmbeddingModel, setShowCustomEmbeddingModel] =
    useState<boolean>(false);
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const [routerConfig, setRouterConfig] = useState<any>(null);

  const initializeForm = useCallback(() => {
    try {
      let parsedConfig = null;
      if (modelData.litellm_params?.auto_router_config) {
        if (typeof modelData.litellm_params.auto_router_config === "string") {
          parsedConfig = JSON.parse(
            modelData.litellm_params.auto_router_config,
          );
        } else {
          parsedConfig = modelData.litellm_params.auto_router_config;
        }
      }

      setRouterConfig(parsedConfig);

      form.setFieldsValue({
        auto_router_name: modelData.model_name,
        auto_router_default_model:
          modelData.litellm_params?.auto_router_default_model || "",
        auto_router_embedding_model:
          modelData.litellm_params?.auto_router_embedding_model || "",
        model_access_group: modelData.model_info?.access_groups || [],
      });

      const allModelGroups = new Set(
        modelInfo.map((model) => model.model_group),
      );
      setShowCustomDefaultModel(
        !allModelGroups.has(modelData.litellm_params?.auto_router_default_model),
      );
      setShowCustomEmbeddingModel(
        !allModelGroups.has(
          modelData.litellm_params?.auto_router_embedding_model,
        ),
      );
    } catch (error) {
      console.error("Error parsing auto router config:", error);
      NotificationsManager.fromBackend("Error loading auto router configuration");
    }
  }, [form, modelData, modelInfo]);

  useEffect(() => {
    if (isVisible && modelData) {
      initializeForm();
    }
  }, [isVisible, modelData, initializeForm]);

  useEffect(() => {
    const fetchModelAccessGroups = async () => {
      if (!accessToken) return;
      try {
        const response = await modelAvailableCall(
          accessToken,
          "",
          "",
          false,
          null,
          true,
          true,
        );
        setModelAccessGroups(
          // eslint-disable-next-line @typescript-eslint/no-explicit-any
          response["data"].map((model: any) => model["id"]),
        );
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
    <Dialog
      open={isVisible}
      onOpenChange={(o) => (!o ? onCancel() : undefined)}
    >
      <DialogContent className="max-w-[1000px]">
        <DialogHeader>
          <DialogTitle>Edit Auto Router Configuration</DialogTitle>
        </DialogHeader>
        <div className="space-y-6">
          <p className="text-muted-foreground">
            Edit the auto router configuration including routing logic, default
            models, and access settings.
          </p>

          <Form form={form} layout="vertical" className="space-y-4">
            <Form.Item
              label="Auto Router Name"
              name="auto_router_name"
              rules={[
                { required: true, message: "Auto router name is required" },
              ]}
            >
              <Input placeholder="e.g., auto_router_1, smart_routing" />
            </Form.Item>

            <div className="w-full">
              <RouterConfigBuilder
                modelInfo={modelInfo}
                value={routerConfig}
                onChange={(config) => {
                  setRouterConfig(config);
                }}
              />
            </div>

            <Form.Item
              label="Default Model"
              name="auto_router_default_model"
              rules={[
                { required: true, message: "Default model is required" },
              ]}
            >
              <AntdSelect
                placeholder="Select a default model"
                onChange={(value) => {
                  setShowCustomDefaultModel(value === "custom");
                }}
                options={[
                  ...modelOptions,
                  { value: "custom", label: "Enter custom model name" },
                ]}
                showSearch={true}
              />
            </Form.Item>

            <Form.Item
              label="Embedding Model"
              name="auto_router_embedding_model"
            >
              <AntdSelect
                placeholder="Select an embedding model (optional)"
                onChange={(value) => {
                  setShowCustomEmbeddingModel(value === "custom");
                }}
                options={[
                  ...modelOptions,
                  { value: "custom", label: "Enter custom model name" },
                ]}
                showSearch={true}
                allowClear
              />
            </Form.Item>

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
        <DialogFooter>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
          <Button onClick={handleSubmit} disabled={loading}>
            {loading ? "Saving..." : "Save Changes"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
};

export default EditAutoRouterModal;
