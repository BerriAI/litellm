"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Button, Form, Input } from "antd";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { getCacheSettingsCall, testCacheConnectionCall, updateCacheSettingsCall } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { buildCachePayload, CacheConnectionInput } from "./helpers";

interface PromptCachingTabProps {
  accessToken: string | null;
}

interface CacheSettingsResponse {
  current_values?: Record<string, unknown>;
}

const stringField = (values: Record<string, unknown>, key: string): string => {
  const value = values[key];
  return value === undefined || value === null ? "" : String(value);
};

const PromptCachingTab: React.FC<PromptCachingTabProps> = ({ accessToken }) => {
  const [form] = Form.useForm<CacheConnectionInput>();
  const [cacheType, setCacheType] = useState<string>("");
  const [isTesting, setIsTesting] = useState<boolean>(false);
  const [isSaving, setIsSaving] = useState<boolean>(false);

  const loadCacheSettings = useCallback(() => {
    if (!accessToken) {
      return;
    }
    getCacheSettingsCall(accessToken)
      .then((data) => {
        const currentValues = (data as CacheSettingsResponse).current_values ?? {};
        setCacheType(stringField(currentValues, "type"));
        form.setFieldsValue({
          host: stringField(currentValues, "host"),
          port: stringField(currentValues, "port"),
          password: "",
        });
      })
      .catch((error) => {
        console.error("Failed to load cache settings:", error);
        NotificationsManager.fromBackend("Failed to load cache settings");
      });
  }, [accessToken, form]);

  useEffect(() => {
    loadCacheSettings();
  }, [loadCacheSettings]);

  const validate = async (): Promise<CacheConnectionInput | null> => {
    try {
      return await form.validateFields();
    } catch {
      return null;
    }
  };

  const handleTest = async () => {
    if (!accessToken) {
      return;
    }
    const values = await validate();
    if (values === null) {
      return;
    }
    setIsTesting(true);
    try {
      const result = await testCacheConnectionCall(accessToken, buildCachePayload(values));
      if (result.status === "success") {
        NotificationsManager.success("Cache connection test successful");
      } else {
        NotificationsManager.fromBackend(`Connection test failed: ${result.message || result.error}`);
      }
    } catch (error) {
      console.error("Test connection error:", error);
      NotificationsManager.fromBackend("Connection test failed");
    } finally {
      setIsTesting(false);
    }
  };

  const handleSave = async () => {
    if (!accessToken) {
      return;
    }
    const values = await validate();
    if (values === null) {
      return;
    }
    setIsSaving(true);
    try {
      await updateCacheSettingsCall(accessToken, buildCachePayload(values));
      NotificationsManager.success("Prompt caching enabled");
      loadCacheSettings();
    } catch (error) {
      console.error("Failed to save cache settings:", error);
      NotificationsManager.fromBackend("Failed to update cache settings");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="w-full space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Prompt caching status</CardTitle>
        </CardHeader>
        <CardContent>
          <span
            className={`rounded-full px-2 py-0.5 text-xs font-medium ${
              cacheType ? "bg-emerald-100 text-emerald-800" : "bg-gray-100 text-gray-600"
            }`}
          >
            {cacheType ? `Enabled (${cacheType})` : "Not configured"}
          </span>
          <p className="mt-2 text-sm text-muted-foreground">
            Global prompt caching reuses identical requests from Redis so repeat calls skip the model and cost nothing
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Redis connection</CardTitle>
        </CardHeader>
        <CardContent>
          <Form form={form} layout="vertical" requiredMark={false}>
            <Form.Item name="host" label="Host" rules={[{ required: true, message: "Host is required" }]}>
              <Input placeholder="my-redis-host" />
            </Form.Item>
            <Form.Item name="port" label="Port" rules={[{ required: true, message: "Port is required" }]}>
              <Input placeholder="6379" />
            </Form.Item>
            <Form.Item name="password" label="Password">
              <Input.Password placeholder="Leave blank to keep existing" />
            </Form.Item>
          </Form>
          <div className="flex justify-end gap-3">
            <Button onClick={handleTest} loading={isTesting}>
              Test connection
            </Button>
            <Button type="primary" onClick={handleSave} loading={isSaving}>
              Enable caching
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
};

export default PromptCachingTab;
