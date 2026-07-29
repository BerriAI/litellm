import React, { useEffect, useMemo, useState } from "react";
import { Button, Form } from "antd";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { StatusBadge } from "@/components/shared/table_cells/status_badge";
import {
  useCoordinationRedisSettings,
  useTestCoordinationRedisConnection,
  useUpdateCoordinationRedisSettings,
} from "@/app/(dashboard)/hooks/coordinationRedis/useCoordinationRedisSettings";
import CoordinationRedisFieldSection from "./CoordinationRedisFieldSection";
import CoordinationRedisTypeSelector from "./CoordinationRedisTypeSelector";
import { CoordinationRedisType } from "./coordinationRedisFields";
import {
  buildCoordinationPayload,
  buildInitialValues,
  configuredSecretFields,
  CoordinationFormValues,
  inferRedisType,
  sourceBadge,
} from "./coordinationRedisUtils";

const CoordinationRedisSettings: React.FC = () => {
  const [form] = Form.useForm<CoordinationFormValues>();
  const [selectedRedisType, setSelectedRedisType] = useState<CoordinationRedisType | null>(null);

  const { data, isLoading, isError } = useCoordinationRedisSettings();
  const updateSettings = useUpdateCoordinationRedisSettings();
  const testConnection = useTestCoordinationRedisConnection();

  const redisType = selectedRedisType ?? inferRedisType(data?.values ?? {});

  useEffect(() => {
    if (data) {
      form.setFieldsValue(buildInitialValues(data.values));
    }
  }, [data, form]);

  useEffect(() => {
    if (isError) {
      NotificationsManager.fromBackend("Failed to load coordination Redis settings");
    }
  }, [isError]);

  const validate = async (): Promise<CoordinationFormValues | null> => {
    try {
      return await form.validateFields();
    } catch {
      return null;
    }
  };

  const handleTestConnection = async () => {
    const values = await validate();
    if (values === null) {
      return;
    }

    try {
      const result = await testConnection.mutateAsync(buildCoordinationPayload(redisType, values));
      if (result.status === "healthy") {
        NotificationsManager.success("Coordination Redis connection test successful!");
      } else {
        NotificationsManager.fromBackend(`Connection test failed: ${result.error ?? "Unknown error"}`);
      }
    } catch (error) {
      NotificationsManager.fromBackend(
        `Connection test failed: ${error instanceof Error ? error.message : "Unknown error"}`,
      );
    }
  };

  const handleSaveChanges = async () => {
    const values = await validate();
    if (values === null) {
      return;
    }

    try {
      await updateSettings.mutateAsync(buildCoordinationPayload(redisType, values));
      NotificationsManager.success("Coordination Redis settings saved. Restart the proxy to apply them.");
    } catch {
      NotificationsManager.fromBackend("Failed to update coordination Redis settings");
    }
  };

  const badge = sourceBadge(data?.source);
  const configuredSecrets = useMemo(() => configuredSecretFields(data?.values ?? {}), [data]);

  return (
    <div className="w-full space-y-8 py-2">
      <Form form={form} layout="vertical" requiredMark={false} className="space-y-6">
        <div className="max-w-3xl space-y-2">
          <div className="flex items-center gap-3">
            <h3 className="text-sm font-medium text-gray-900">Coordination Redis</h3>
            {!isLoading && <StatusBadge tone={badge.tone} label={badge.label} dataTestId="coordination-redis-source" />}
          </div>
          <p className="text-xs text-gray-500">
            Redis used to coordinate work across proxy pods: cross-pod rate limits, spend tracking, and the pod lock
            manager. It is configured independently of the response cache.
          </p>
          <p className="text-xs text-gray-500">{badge.tooltip}</p>
          <p className="text-xs text-amber-600">Saved changes take effect on proxy restart.</p>
        </div>

        <CoordinationRedisTypeSelector redisType={redisType} onTypeChange={setSelectedRedisType} />

        <div className="pt-4 border-t border-gray-200">
          <CoordinationRedisFieldSection
            title="Connection Settings"
            section="connection"
            redisType={redisType}
            configuredSecrets={configuredSecrets}
          />
        </div>

        {redisType === "cluster" && (
          <div className="pt-4 border-t border-gray-200">
            <CoordinationRedisFieldSection
              title="Cluster Configuration"
              section="cluster"
              redisType={redisType}
              configuredSecrets={configuredSecrets}
              gridCols="grid-cols-1 gap-6"
            />
          </div>
        )}

        {redisType === "sentinel" && (
          <div className="pt-4 border-t border-gray-200">
            <CoordinationRedisFieldSection
              title="Sentinel Configuration"
              section="sentinel"
              redisType={redisType}
              configuredSecrets={configuredSecrets}
            />
          </div>
        )}

        <div className="pt-4 border-t border-gray-200">
          <CoordinationRedisFieldSection
            title="SSL Settings"
            section="ssl"
            redisType={redisType}
            configuredSecrets={configuredSecrets}
          />
        </div>
      </Form>

      <div className="border-t border-gray-200 pt-6 flex justify-end gap-3">
        <Button onClick={handleTestConnection} loading={testConnection.isPending}>
          {testConnection.isPending ? "Testing..." : "Test Connection"}
        </Button>
        <Button type="primary" onClick={handleSaveChanges} loading={updateSettings.isPending}>
          {updateSettings.isPending ? "Saving..." : "Save Changes"}
        </Button>
      </div>
    </div>
  );
};

export default CoordinationRedisSettings;
