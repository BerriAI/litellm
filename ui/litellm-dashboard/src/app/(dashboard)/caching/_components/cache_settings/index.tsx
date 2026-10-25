import { useLanguage } from "@/contexts/LanguageContext";
import React, { useState, useEffect, useCallback } from "react";
import TranslatedMessage from "@/components/shared/TranslatedMessage";
import { ChevronRight } from "lucide-react";
import { FormProvider, useForm } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { getCacheSettingsCall, testCacheConnectionCall, updateCacheSettingsCall } from "@/components/networking";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import { toast } from "@/lib/toast";
import RedisTypeSelector from "./RedisTypeSelector";
import CacheFieldSection from "./CacheFieldSection";
import { EmbeddingModelOption } from "./CacheFormField";
import { CACHE_FIELDS, REDIS_TYPES, REDIS_TYPE_DESCRIPTIONS, RedisType } from "./cacheSettingsFields";
import {
  buildCachePayload,
  buildInitialValues,
  CacheFormValues,
  configuredSecretFields,
  isFieldVisible,
} from "./cacheSettingsUtils";

const ADVANCED_SECTIONS = ["ssl", "cacheManagement", "gcp"] as const;

interface CacheSettingsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

const toRedisType = (value: unknown): RedisType =>
  REDIS_TYPES.includes(value as RedisType) ? (value as RedisType) : "node";

const CacheSettings: React.FC<CacheSettingsProps> = ({ accessToken }) => {
  const { t } = useLanguage();
  const form = useForm<CacheFormValues>({ defaultValues: buildInitialValues({}) });
  const [redisType, setRedisType] = useState<RedisType>("node");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [embeddingModels, setEmbeddingModels] = useState<EmbeddingModelOption[]>([]);
  const [isTesting, setIsTesting] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [configuredSecrets, setConfiguredSecrets] = useState<ReadonlySet<string>>(new Set());

  const loadCacheSettings = useCallback(async () => {
    if (!accessToken) {
      return;
    }
    try {
      const data = (await getCacheSettingsCall(accessToken)) as { current_values?: Record<string, unknown> };
      const currentValues = data.current_values ?? {};
      form.reset(buildInitialValues(currentValues));
      setConfiguredSecrets(configuredSecretFields(currentValues));
      setRedisType(toRedisType(currentValues.redis_type));
    } catch (error) {
      console.error("Failed to load cache settings:", error);
      // 本地通知在展示时翻译，切换语言不重发请求或重置配置草稿。
      toast.fromError("Failed to load cache settings", {
        description: <TranslatedMessage message="Failed to load cache settings" />,
      });
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

  const validate = (): CacheFormValues | null => {
    const values = form.getValues();
    const failures = CACHE_FIELDS.filter(
      (field) =>
        isFieldVisible(field, redisType) &&
        (advancedOpen || !ADVANCED_SECTIONS.some((section) => section === field.section)),
    ).flatMap((field) => {
      const message = field.rules?.map((rule) => rule(values[field.name])).find((result) => result !== null);
      return message === undefined || message === null ? [] : [[field.name, message] as const];
    });

    form.clearErrors();
    failures.forEach(([name, message]) => form.setError(name, { type: "local-validation", message }));
    return failures.length > 0 ? null : values;
  };

  const handleTestConnection = async () => {
    if (!accessToken) {
      return;
    }
    const values = validate();
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
        toast.success(<TranslatedMessage message="Cache connection test successful!" />);
      } else {
        const detail = result.message || result.error;
        toast.fromError(`Connection test failed: ${detail ?? "Unknown error"}`, {
          description: (
            <TranslatedMessage
              message={detail == null ? "Connection test failed: Unknown error" : "Connection test failed: {error}"}
              values={{ error: String(detail) }}
            />
          ),
        });
      }
    } catch (error) {
      console.error("Test connection error:", error);
      toast.fromError(`Connection test failed: ${error instanceof Error ? error.message : "Unknown error"}`, {
        description: (
          <TranslatedMessage
            message={
              error instanceof Error ? "Connection test failed: {error}" : "Connection test failed: Unknown error"
            }
            values={{ error: error instanceof Error ? error.message : "" }}
          />
        ),
      });
    } finally {
      setIsTesting(false);
    }
  };

  const handleSaveChanges = async () => {
    if (!accessToken) {
      return;
    }
    const values = validate();
    if (values === null) {
      return;
    }

    setIsSaving(true);
    try {
      await updateCacheSettingsCall(accessToken, buildCachePayload(redisType, values, { forTesting: false }));
      toast.success(<TranslatedMessage message="Cache settings updated successfully" />);
      await loadCacheSettings();
    } catch (error) {
      console.error("Failed to save cache settings:", error);
      toast.fromError("Failed to update cache settings", {
        description: <TranslatedMessage message="Failed to update cache settings" />,
      });
    } finally {
      setIsSaving(false);
    }
  };

  if (!accessToken) {
    return null;
  }

  return (
    <div className="w-full space-y-8 py-2">
      <FormProvider {...form}>
        <form onSubmit={(event) => event.preventDefault()} className="space-y-6">
          <div className="max-w-3xl">
            <h3 className="text-sm font-medium text-foreground">{t("Cache Settings")}</h3>
            <p className="text-xs text-muted-foreground mt-1">{t("Configure Redis cache for LiteLLM")}</p>
          </div>

          <RedisTypeSelector
            redisType={redisType}
            redisTypeDescriptions={REDIS_TYPE_DESCRIPTIONS}
            onTypeChange={(type) => setRedisType(toRedisType(type))}
          />

          <div className="pt-4 border-t border-border">
            <CacheFieldSection
              title={t("Connection Settings")}
              section="connection"
              redisType={redisType}
              embeddingModels={embeddingModels}
              configuredSecrets={configuredSecrets}
            />
          </div>

          {redisType === "cluster" && (
            <div className="pt-4 border-t border-border">
              <CacheFieldSection
                title={t("Cluster Configuration")}
                section="cluster"
                redisType={redisType}
                embeddingModels={embeddingModels}
                gridCols="grid-cols-1 gap-6"
              />
            </div>
          )}

          {redisType === "sentinel" && (
            <div className="pt-4 border-t border-border">
              <CacheFieldSection
                title={t("Sentinel Configuration")}
                section="sentinel"
                redisType={redisType}
                embeddingModels={embeddingModels}
                configuredSecrets={configuredSecrets}
              />
            </div>
          )}

          {redisType === "semantic" && (
            <div className="pt-4 border-t border-border">
              <CacheFieldSection
                title={t("Semantic Configuration")}
                section="semantic"
                redisType={redisType}
                embeddingModels={embeddingModels}
              />
            </div>
          )}

          <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen} className="mt-4">
            <CollapsibleTrigger className="group flex w-full items-center justify-between py-2 text-left">
              <span className="text-sm font-medium text-foreground">{t("Advanced Settings")}</span>
              <ChevronRight className="size-4 text-muted-foreground transition-transform group-data-panel-open:rotate-90" />
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="space-y-6">
                <CacheFieldSection
                  title={t("SSL Settings")}
                  section="ssl"
                  redisType={redisType}
                  embeddingModels={embeddingModels}
                  headingLevel="h5"
                />
                <CacheFieldSection
                  title={t("Cache Management")}
                  section="cacheManagement"
                  redisType={redisType}
                  embeddingModels={embeddingModels}
                  headingLevel="h5"
                />
                <CacheFieldSection
                  title={t("GCP Authentication")}
                  section="gcp"
                  redisType={redisType}
                  embeddingModels={embeddingModels}
                  headingLevel="h5"
                />
              </div>
            </CollapsibleContent>
          </Collapsible>
        </form>
      </FormProvider>

      <div className="border-t border-border pt-6 flex justify-end gap-3">
        <Button variant="secondary" size="sm" onClick={handleTestConnection} disabled={isTesting} className="text-sm">
          {isTesting ? t("Testing...") : t("Test Connection")}
        </Button>
        <Button size="sm" onClick={handleSaveChanges} disabled={isSaving} className="text-sm font-medium">
          {isSaving ? t("Saving...") : t("Save Changes")}
        </Button>
      </div>
    </div>
  );
};

export default CacheSettings;
