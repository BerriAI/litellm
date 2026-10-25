import { useLanguage } from "@/contexts/LanguageContext";
import React, { useEffect, useMemo, useState } from "react";
import TranslatedMessage from "@/components/shared/TranslatedMessage";
import { FormProvider, useForm } from "react-hook-form";
import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { toast } from "@/lib/toast";
import { StatusBadge } from "@/components/shared/table_cells/status_badge";
import {
  useCoordinationRedisSettings,
  useTestCoordinationRedisConnection,
  useUpdateCoordinationRedisSettings,
} from "@/app/(dashboard)/hooks/coordinationRedis/useCoordinationRedisSettings";
import CoordinationRedisFieldSection from "./CoordinationRedisFieldSection";
import CoordinationRedisTypeSelector from "./CoordinationRedisTypeSelector";
import { COORDINATION_FIELDS, CoordinationRedisType } from "./coordinationRedisFields";
import {
  buildCoordinationPayload,
  buildInitialValues,
  configuredSecretFields,
  CoordinationFormValues,
  inferRedisType,
  isFieldVisible,
  sourceBadge,
} from "./coordinationRedisUtils";

const CoordinationRedisSettings: React.FC = () => {
  const { t } = useLanguage();
  const form = useForm<CoordinationFormValues>({ defaultValues: buildInitialValues({}) });
  const [selectedRedisType, setSelectedRedisType] = useState<CoordinationRedisType | null>(null);

  const { data, isLoading, isError } = useCoordinationRedisSettings();
  const updateSettings = useUpdateCoordinationRedisSettings();
  const testConnection = useTestCoordinationRedisConnection();

  const redisType = selectedRedisType ?? inferRedisType(data?.values ?? {});

  useEffect(() => {
    if (data) {
      form.reset(buildInitialValues(data.values));
    }
  }, [data, form]);

  useEffect(() => {
    if (isError) {
      // 通知只跟随错误状态发出，已显示的文案由语言上下文更新。
      toast.fromError("Failed to load coordination Redis settings", {
        description: <TranslatedMessage message="Failed to load coordination Redis settings" />,
      });
    }
  }, [isError]);

  const validate = (): CoordinationFormValues | null => {
    const values = form.getValues();
    const failures = COORDINATION_FIELDS.filter((field) => isFieldVisible(field, redisType)).flatMap((field) => {
      const message = field.rules?.map((rule) => rule(values[field.name])).find((result) => result !== null);
      return message === undefined || message === null ? [] : [[field.name, message] as const];
    });

    form.clearErrors();
    failures.forEach(([name, message]) => form.setError(name, { type: "local-validation", message }));
    return failures.length > 0 ? null : values;
  };

  const handleTestConnection = async () => {
    const values = validate();
    if (values === null) {
      return;
    }

    try {
      const result = await testConnection.mutateAsync(buildCoordinationPayload(redisType, values));
      if (result.status === "healthy") {
        toast.success(<TranslatedMessage message="Coordination Redis connection test successful!" />);
      } else {
        toast.fromError(`Connection test failed: ${result.error ?? "Unknown error"}`, {
          description: (
            <TranslatedMessage
              message={
                result.error == null ? "Connection test failed: Unknown error" : "Connection test failed: {error}"
              }
              values={{ error: result.error ?? "" }}
            />
          ),
        });
      }
    } catch (error) {
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
    }
  };

  const handleSaveChanges = async () => {
    const values = validate();
    if (values === null) {
      return;
    }

    try {
      await updateSettings.mutateAsync(buildCoordinationPayload(redisType, values));
      toast.success(
        <TranslatedMessage message="Coordination Redis settings saved. Restart the proxy to apply them." />,
      );
    } catch {
      toast.fromError("Failed to update coordination Redis settings", {
        description: <TranslatedMessage message="Failed to update coordination Redis settings" />,
      });
    }
  };

  const badge = sourceBadge(data?.source);
  const configuredSecrets = useMemo(() => configuredSecretFields(data?.values ?? {}), [data]);

  return (
    <div className="w-full space-y-8 py-2">
      <FormProvider {...form}>
        <form onSubmit={(event) => event.preventDefault()} className="space-y-6">
          <div className="max-w-3xl space-y-2">
            <div className="flex items-center gap-3">
              <h3 className="text-sm font-medium text-foreground">{t("Coordination Redis")}</h3>
              {!isLoading && (
                <StatusBadge tone={badge.tone} label={t(badge.label)} dataTestId="coordination-redis-source" />
              )}
            </div>
            <p className="text-xs text-muted-foreground">
              {t(
                "Redis used to coordinate work across proxy pods: cross-pod rate limits, spend tracking, and the pod lock manager. It is configured independently of the response cache.",
              )}
            </p>
            <p className="text-xs text-muted-foreground">{t(badge.tooltip)}</p>
            <p className="text-xs text-warning">{t("Saved changes take effect on proxy restart.")}</p>
          </div>

          <CoordinationRedisTypeSelector redisType={redisType} onTypeChange={setSelectedRedisType} />

          <div className="pt-4 border-t border-border">
            <CoordinationRedisFieldSection
              title={t("Connection Settings")}
              section="connection"
              redisType={redisType}
              configuredSecrets={configuredSecrets}
            />
          </div>

          {redisType === "cluster" && (
            <div className="pt-4 border-t border-border">
              <CoordinationRedisFieldSection
                title={t("Cluster Configuration")}
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
                title={t("Sentinel Configuration")}
                section="sentinel"
                redisType={redisType}
                configuredSecrets={configuredSecrets}
              />
            </div>
          )}

          <div className="pt-4 border-t border-border">
            <CoordinationRedisFieldSection
              title={t("SSL Settings")}
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
          {testConnection.isPending ? t("Testing...") : t("Test Connection")}
        </Button>
        <Button onClick={handleSaveChanges} disabled={updateSettings.isPending}>
          {updateSettings.isPending && <UiLoadingSpinner className="size-4" />}
          {updateSettings.isPending ? t("Saving...") : t("Save Changes")}
        </Button>
      </div>
    </div>
  );
};

export default CoordinationRedisSettings;
