import React, { useState, useEffect, useCallback } from "react";
import { ChevronRight } from "lucide-react";
import { FormProvider } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { getCacheSettingsCall, testCacheConnectionCall, updateCacheSettingsCall } from "@/components/networking";
import { fetchAvailableModels, ModelGroup } from "@/components/llm_calls/fetch_models";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";
import RedisTypeSelector from "./RedisTypeSelector";
import CacheFieldSection from "./CacheFieldSection";
import { EmbeddingModelOption } from "./CacheFormField";
import { REDIS_TYPES, REDIS_TYPE_DESCRIPTIONS, RedisType } from "./cacheSettingsFields";
import { buildCachePayload, buildInitialValues, configuredSecretFields } from "./mapper";
import { createCacheSettingsSchema } from "./schema";

interface CacheSettingsProps {
  accessToken: string | null;
  userRole: string | null;
  userID: string | null;
}

const toRedisType = (value: unknown): RedisType =>
  REDIS_TYPES.includes(value as RedisType) ? (value as RedisType) : "node";

const CacheSettings: React.FC<CacheSettingsProps> = ({ accessToken }) => {
  const { t } = useTranslation();
  const [redisType, setRedisType] = useState<RedisType>("node");
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const schema = createCacheSettingsSchema(redisType, advancedOpen, t);
  const form = useZodForm(schema, { defaultValues: buildInitialValues({}) });
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
      toast.fromError(t("caching.settings.toasts.loadFailed"));
    }
  }, [accessToken, form, t]);

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

  const validate = () => {
    const values = form.getValues();
    const result = schema.safeParse(values);
    form.clearErrors();
    if (result.success) return values;

    for (const issue of result.error.issues) {
      const [name] = issue.path;
      if (typeof name === "string") form.setError(name, { message: issue.message });
    }
    return null;
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
        toast.success(t("caching.settings.toasts.testSucceeded"));
      } else {
        toast.fromError(t("caching.settings.toasts.testFailed", { error: String(result.message || result.error) }));
      }
    } catch (error) {
      console.error("Test connection error:", error);
      toast.fromError(
        t("caching.settings.toasts.testFailed", {
          error: error instanceof Error ? error.message : t("caching.settings.toasts.unknownError"),
        }),
      );
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
      toast.success(t("caching.settings.toasts.saved"));
      await loadCacheSettings();
    } catch (error) {
      console.error("Failed to save cache settings:", error);
      toast.fromError(t("caching.settings.toasts.saveFailed"));
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
            <h3 className="text-sm font-medium text-foreground">{t("caching.settings.title")}</h3>
            <p className="text-xs text-muted-foreground mt-1">{t("caching.settings.subtitle")}</p>
          </div>

          <RedisTypeSelector
            redisType={redisType}
            redisTypeDescriptions={REDIS_TYPE_DESCRIPTIONS}
            onTypeChange={(type) => setRedisType(toRedisType(type))}
          />

          <div className="pt-4 border-t border-border">
            <CacheFieldSection
              title={t("caching.settings.sections.connection")}
              section="connection"
              redisType={redisType}
              embeddingModels={embeddingModels}
              configuredSecrets={configuredSecrets}
            />
          </div>

          {redisType === "cluster" && (
            <div className="pt-4 border-t border-border">
              <CacheFieldSection
                title={t("caching.settings.sections.cluster")}
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
                title={t("caching.settings.sections.sentinel")}
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
                title={t("caching.settings.sections.semantic")}
                section="semantic"
                redisType={redisType}
                embeddingModels={embeddingModels}
              />
            </div>
          )}

          <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen} className="mt-4">
            <CollapsibleTrigger className="group flex w-full items-center justify-between py-2 text-left">
              <span className="text-sm font-medium text-foreground">{t("caching.settings.advanced")}</span>
              <ChevronRight className="size-4 text-muted-foreground transition-transform group-data-panel-open:rotate-90" />
            </CollapsibleTrigger>
            <CollapsibleContent>
              <div className="space-y-6">
                <CacheFieldSection
                  title={t("caching.settings.sections.ssl")}
                  section="ssl"
                  redisType={redisType}
                  embeddingModels={embeddingModels}
                  headingLevel="h5"
                />
                <CacheFieldSection
                  title={t("caching.settings.sections.cacheManagement")}
                  section="cacheManagement"
                  redisType={redisType}
                  embeddingModels={embeddingModels}
                  headingLevel="h5"
                />
                <CacheFieldSection
                  title={t("caching.settings.sections.gcp")}
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
          {isTesting ? t("caching.settings.testing") : t("caching.settings.testConnection")}
        </Button>
        <Button size="sm" onClick={handleSaveChanges} disabled={isSaving} className="text-sm font-medium">
          {isSaving ? t("caching.settings.saving") : t("caching.settings.saveChanges")}
        </Button>
      </div>
    </div>
  );
};

export default CacheSettings;
