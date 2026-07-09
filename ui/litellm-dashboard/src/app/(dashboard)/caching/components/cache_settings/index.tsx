import React, { useState, useEffect, useCallback } from "react";
import { Button, Accordion, AccordionHeader, AccordionBody } from "@tremor/react";
import { Form } from "antd";
import { getCacheSettingsCall, testCacheConnectionCall, updateCacheSettingsCall } from "@/components/networking";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import NotificationsManager from "@/components/molecules/notifications_manager";
import CacheTypeSelector from "./CacheTypeSelector";
import CacheFieldSection from "./CacheFieldSection";
import { EmbeddingModelOption } from "./CacheFormField";
import { CacheMode, REDIS_DEPLOYMENT_TYPES, RedisDeploymentType, RedisType } from "./cacheSettingsFields";
import { buildCachePayload, buildInitialValues, CacheFormValues } from "./cacheSettingsUtils";

interface CacheSettingsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

interface CacheSelection {
  cacheMode: CacheMode;
  deploymentType: RedisDeploymentType;
}

const toCacheSelection = (value: unknown): CacheSelection => {
  if (value === "semantic") {
    return { cacheMode: "semantic", deploymentType: "node" };
  }
  const deploymentType = REDIS_DEPLOYMENT_TYPES.includes(value as RedisDeploymentType)
    ? (value as RedisDeploymentType)
    : "node";
  return { cacheMode: "standard", deploymentType };
};

const CacheSettings: React.FC<CacheSettingsProps> = ({ accessToken }) => {
  const [form] = Form.useForm<CacheFormValues>();
  const [cacheMode, setCacheMode] = useState<CacheMode>("standard");
  const [deploymentType, setDeploymentType] = useState<RedisDeploymentType>("node");
  const redisType: RedisType = cacheMode === "semantic" ? "semantic" : deploymentType;
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModelOption[]>([]);
  const [isTesting, setIsTesting] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);

  const loadCacheSettings = useCallback(async () => {
    if (!accessToken) {
      return;
    }
    try {
      const data = (await getCacheSettingsCall(accessToken)) as { current_values?: Record<string, unknown> };
      const currentValues = data.current_values ?? {};
      form.setFieldsValue(buildInitialValues(currentValues));
      const selection = toCacheSelection(currentValues.redis_type);
      setCacheMode(selection.cacheMode);
      setDeploymentType(selection.deploymentType);
    } catch (error) {
      console.error("Failed to load cache settings:", error);
      NotificationsManager.fromBackend("Failed to load cache settings");
    }
  }, [accessToken, form]);

  useEffect(() => {
    loadCacheSettings();
  }, [loadCacheSettings]);

  useEffect(() => {
    if (!accessToken) {
      return;
    }
    fetchAvailableModels(accessToken)
      .then((models: ModelGroup[]) =>
        setEmbeddingModels(
          models
            .filter((model) => model.mode === "embedding")
            .map((model) => ({ value: model.model_group, label: model.model_group })),
        ),
      )
      .catch((error) => console.error("Error fetching embedding models:", error));
  }, [accessToken]);

  const validate = async (): Promise<CacheFormValues | null> => {
    try {
      return await form.validateFields();
    } catch {
      return null;
    }
  };

  const handleTestConnection = async () => {
    if (!accessToken) {
      return;
    }
    const values = await validate();
    if (values === null) {
      return;
    }

    setIsTesting(true);
    try {
      const result = await testCacheConnectionCall(
        accessToken,
        buildCachePayload(redisType, values, { forTesting: true }),
      );
      if (result.status === "success") {
        NotificationsManager.success("Cache connection test successful!");
      } else {
        NotificationsManager.fromBackend(`Connection test failed: ${result.message || result.error}`);
      }
    } catch (error) {
      console.error("Test connection error:", error);
      NotificationsManager.fromBackend(
        `Connection test failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
    } finally {
      setIsTesting(false);
    }
  };

  const handleSaveChanges = async () => {
    if (!accessToken) {
      return;
    }
    const values = await validate();
    if (values === null) {
      return;
    }

    setIsSaving(true);
    try {
      await updateCacheSettingsCall(accessToken, buildCachePayload(redisType, values, { forTesting: false }));
      NotificationsManager.success("Cache settings updated successfully");
      await loadCacheSettings();
    } catch (error) {
      console.error("Failed to save cache settings:", error);
      NotificationsManager.fromBackend("Failed to update cache settings");
    } finally {
      setIsSaving(false);
    }
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full space-y-8 py-2">
      <Form form={form} layout="vertical" requiredMark={false} className="space-y-6">
        <div className="max-w-3xl">
          <h3 className="text-sm font-medium text-gray-900">Cache Settings</h3>
          <p className="text-xs text-gray-500 mt-1">Configure Redis cache for LiteLLM</p>
        </div>

        <CacheTypeSelector
          cacheMode={cacheMode}
          deploymentType={deploymentType}
          onCacheModeChange={setCacheMode}
          onDeploymentTypeChange={setDeploymentType}
        />

        <div className="pt-4 border-t border-gray-200">
          <CacheFieldSection
            title="Connection Settings"
            section="connection"
            redisType={redisType}
            embeddingModels={embeddingModels}
          />
        </div>

        {redisType === "cluster" && (
          <div className="pt-4 border-t border-gray-200">
            <CacheFieldSection
              title="Cluster Configuration"
              section="cluster"
              redisType={redisType}
              embeddingModels={embeddingModels}
              gridCols="grid-cols-1 gap-6"
            />
          </div>
        )}

        {redisType === "sentinel" && (
          <div className="pt-4 border-t border-gray-200">
            <CacheFieldSection
              title="Sentinel Configuration"
              section="sentinel"
              redisType={redisType}
              embeddingModels={embeddingModels}
            />
          </div>
        )}

        {redisType === "semantic" && (
          <div className="pt-4 border-t border-gray-200">
            <CacheFieldSection
              title="Semantic Configuration"
              section="semantic"
              redisType={redisType}
              embeddingModels={embeddingModels}
            />
          </div>
        )}

        <Accordion className="mt-4">
          <AccordionHeader>
            <span className="text-sm font-medium text-gray-900">Advanced Settings</span>
          </AccordionHeader>
          <AccordionBody>
            <div className="space-y-6">
              <CacheFieldSection
                title="SSL Settings"
                section="ssl"
                redisType={redisType}
                embeddingModels={embeddingModels}
                headingLevel="h5"
              />
              <CacheFieldSection
                title="Cache Management"
                section="cacheManagement"
                redisType={redisType}
                embeddingModels={embeddingModels}
                headingLevel="h5"
              />
              <CacheFieldSection
                title="GCP Authentication"
                section="gcp"
                redisType={redisType}
                embeddingModels={embeddingModels}
                headingLevel="h5"
              />
            </div>
          </AccordionBody>
        </Accordion>
      </Form>

      <div className="border-t border-gray-200 pt-6 flex justify-end gap-3">
        <Button variant="secondary" size="sm" onClick={handleTestConnection} disabled={isTesting} className="text-sm">
          {isTesting ? "Testing..." : "Test Connection"}
        </Button>
        <Button size="sm" onClick={handleSaveChanges} disabled={isSaving} className="text-sm font-medium">
          {isSaving ? "Saving..." : "Save Changes"}
        </Button>
      </div>
    </div>
  );
};

export default CacheSettings;
