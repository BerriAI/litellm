import React, { useEffect, useMemo, useState } from "react";
import { FormProvider } from "react-hook-form";
import { useTranslation } from "react-i18next";
import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useZodForm } from "@/lib/forms/useZodForm";
import { toast } from "@/lib/toast";
import { StatusBadge } from "@/components/shared/table_cells/status_badge";
import {
  useCoordinationRedisSettings,
  useTestCoordinationRedisConnection,
  useUpdateCoordinationRedisSettings,
} from "@/app/(dashboard)/hooks/coordinationRedis/useCoordinationRedisSettings";
import CoordinationRedisFieldSection from "./CoordinationRedisFieldSection";
import CoordinationRedisTypeSelector from "./CoordinationRedisTypeSelector";
import type { CoordinationRedisType } from "./coordinationRedisFields";
import {
  buildCoordinationPayload,
  buildInitialValues,
  configuredSecretFields,
  inferRedisType,
  sourceBadge,
} from "./mapper";
import { createCoordinationRedisSchema } from "./schema";

const CoordinationRedisSettings: React.FC = () => {
  const { t } = useTranslation();
  const [selectedRedisType, setSelectedRedisType] = useState<CoordinationRedisType | null>(null);

  const { data, isLoading, isError } = useCoordinationRedisSettings();
  const updateSettings = useUpdateCoordinationRedisSettings();
  const testConnection = useTestCoordinationRedisConnection();

  const redisType = selectedRedisType ?? inferRedisType(data?.values ?? {});
  const schema = createCoordinationRedisSchema(redisType, t);
  const form = useZodForm(schema, { defaultValues: buildInitialValues({}) });

  useEffect(() => {
    if (data) {
      form.reset(buildInitialValues(data.values));
    }
  }, [data, form]);

  useEffect(() => {
    if (isError) {
      toast.fromError(t("caching.coordinationRedis.toasts.loadFailed"));
    }
  }, [isError, t]);

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
    const values = validate();
    if (values === null) {
      return;
    }

    try {
      const result = await testConnection.mutateAsync(buildCoordinationPayload(redisType, values));
      if (result.status === "healthy") {
        toast.success(t("caching.coordinationRedis.toasts.testSucceeded"));
      } else {
        toast.fromError(
          t("caching.coordinationRedis.toasts.testFailed", {
            error: result.error ?? t("caching.coordinationRedis.toasts.unknownError"),
          }),
        );
      }
    } catch (error) {
      toast.fromError(
        t("caching.coordinationRedis.toasts.testFailed", {
          error: error instanceof Error ? error.message : t("caching.coordinationRedis.toasts.unknownError"),
        }),
      );
    }
  };

  const handleSaveChanges = async () => {
    const values = validate();
    if (values === null) {
      return;
    }

    try {
      await updateSettings.mutateAsync(buildCoordinationPayload(redisType, values));
      toast.success(t("caching.coordinationRedis.toasts.saved"));
    } catch {
      toast.fromError(t("caching.coordinationRedis.toasts.saveFailed"));
    }
  };

  const badge = sourceBadge(data?.source, t);
  const configuredSecrets = useMemo(() => configuredSecretFields(data?.values ?? {}), [data]);

  return (
    <div className="w-full space-y-8 py-2">
      <FormProvider {...form}>
        <form onSubmit={(event) => event.preventDefault()} className="space-y-6">
          <div className="max-w-3xl space-y-2">
            <div className="flex items-center gap-3">
              <h3 className="text-sm font-medium text-foreground">{t("caching.coordinationRedis.title")}</h3>
              {!isLoading && (
                <StatusBadge tone={badge.tone} label={badge.label} dataTestId="coordination-redis-source" />
              )}
            </div>
            <p className="text-xs text-muted-foreground">{t("caching.coordinationRedis.description")}</p>
            <p className="text-xs text-muted-foreground">{badge.tooltip}</p>
            <p className="text-xs text-warning">{t("caching.coordinationRedis.restartNote")}</p>
          </div>

          <CoordinationRedisTypeSelector redisType={redisType} onTypeChange={setSelectedRedisType} />

          <div className="pt-4 border-t border-border">
            <CoordinationRedisFieldSection
              title={t("caching.coordinationRedis.sections.connection")}
              section="connection"
              redisType={redisType}
              configuredSecrets={configuredSecrets}
            />
          </div>

          {redisType === "cluster" && (
            <div className="pt-4 border-t border-border">
              <CoordinationRedisFieldSection
                title={t("caching.coordinationRedis.sections.cluster")}
                section="cluster"
                redisType={redisType}
                configuredSecrets={configuredSecrets}
                gridCols="grid-cols-1 gap-6"
              />
            </div>
          )}

          {redisType === "sentinel" && (
            <div className="pt-4 border-t border-border">
              <CoordinationRedisFieldSection
                title={t("caching.coordinationRedis.sections.sentinel")}
                section="sentinel"
                redisType={redisType}
                configuredSecrets={configuredSecrets}
              />
            </div>
          )}

          <div className="pt-4 border-t border-border">
            <CoordinationRedisFieldSection
              title={t("caching.coordinationRedis.sections.ssl")}
              section="ssl"
              redisType={redisType}
              configuredSecrets={configuredSecrets}
            />
          </div>
        </form>
      </FormProvider>

      <div className="border-t border-border pt-6 flex justify-end gap-3">
        <Button variant="outline" onClick={handleTestConnection} disabled={testConnection.isPending}>
          {testConnection.isPending && <UiLoadingSpinner className="size-4" />}
          {testConnection.isPending
            ? t("caching.coordinationRedis.testing")
            : t("caching.coordinationRedis.testConnection")}
        </Button>
        <Button onClick={handleSaveChanges} disabled={updateSettings.isPending}>
          {updateSettings.isPending && <UiLoadingSpinner className="size-4" />}
          {updateSettings.isPending
            ? t("caching.coordinationRedis.saving")
            : t("caching.coordinationRedis.saveChanges")}
        </Button>
      </div>
    </div>
  );
};

export default CoordinationRedisSettings;
