"use client";

import React, { useCallback, useEffect, useState } from "react";
import { Button, Form, Input, Switch } from "antd";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { createGuardrailCall, getGuardrailsList } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";
import {
  buildCompressionGuardrailPayload,
  compressionGuardrailsOf,
  GuardrailListItem,
  GuardrailListResponse,
} from "./helpers";

interface PromptCompressionTabProps {
  accessToken: string | null;
}

interface CompressionFormValues {
  name: string;
  apiBase: string;
  defaultOn: boolean;
}

const PromptCompressionTab: React.FC<PromptCompressionTabProps> = ({ accessToken }) => {
  const [form] = Form.useForm<CompressionFormValues>();
  const [guardrails, setGuardrails] = useState<GuardrailListItem[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isSaving, setIsSaving] = useState<boolean>(false);

  const loadGuardrails = useCallback(() => {
    if (!accessToken) {
      return;
    }
    getGuardrailsList(accessToken)
      .then((response) => setGuardrails(compressionGuardrailsOf(response as GuardrailListResponse)))
      .catch((error) => {
        console.error("Failed to load compression guardrails:", error);
        NotificationsManager.fromBackend("Failed to load compression guardrails");
      })
      .finally(() => setIsLoading(false));
  }, [accessToken]);

  useEffect(() => {
    loadGuardrails();
  }, [loadGuardrails]);

  const handleAdd = async (values: CompressionFormValues) => {
    if (!accessToken) {
      return;
    }
    setIsSaving(true);
    try {
      await createGuardrailCall(
        accessToken,
        buildCompressionGuardrailPayload({
          name: values.name,
          apiBase: values.apiBase,
          defaultOn: values.defaultOn ?? true,
        }),
      );
      NotificationsManager.success("Compression guardrail created");
      form.resetFields();
      await loadGuardrails();
    } catch (error) {
      console.error("Failed to create compression guardrail:", error);
      NotificationsManager.fromBackend("Failed to create compression guardrail");
    } finally {
      setIsSaving(false);
    }
  };

  return (
    <div className="w-full space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Headroom prompt compression</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4 text-sm text-muted-foreground">
            Headroom is a native LiteLLM guardrail that compresses your prompts before they reach the model, so you pay
            for fewer input tokens. The tokens it removes are priced and shown on the Usage tab as compression savings.{" "}
            <a
              href="https://docs.litellm.ai/docs/proxy/headroom"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 underline"
            >
              Headroom setup docs
            </a>
          </p>
          {isLoading && <p className="text-sm text-muted-foreground">Loading...</p>}
          {!isLoading && guardrails.length === 0 && (
            <p className="text-sm text-muted-foreground">
              No prompt compression guardrails configured yet. Add one below to start saving on input tokens
            </p>
          )}
          {!isLoading && guardrails.length > 0 && (
            <ul className="divide-y divide-gray-200">
              {guardrails.map((guardrail) => (
                <li key={guardrail.guardrail_id} className="flex items-center justify-between py-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">{guardrail.guardrail_name}</p>
                    <p className="text-xs text-muted-foreground">{guardrail.litellm_params?.api_base ?? ""}</p>
                  </div>
                  <span
                    className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                      guardrail.litellm_params?.default_on
                        ? "bg-emerald-100 text-emerald-800"
                        : "bg-gray-100 text-gray-600"
                    }`}
                  >
                    {guardrail.litellm_params?.default_on ? "Always on" : "Opt-in"}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Add Headroom compression guardrail</CardTitle>
        </CardHeader>
        <CardContent>
          <Form
            form={form}
            layout="vertical"
            requiredMark={false}
            onFinish={handleAdd}
            initialValues={{ defaultOn: true }}
          >
            <Form.Item name="name" label="Name" rules={[{ required: true, message: "Name is required" }]}>
              <Input placeholder="headroom-compression" />
            </Form.Item>
            <Form.Item
              name="apiBase"
              label="Headroom API base"
              tooltip="Base URL of your Headroom compression service (LiteLLM calls its /v1/compress endpoint)"
              extra="The URL where your Headroom compression service is hosted"
              rules={[{ required: true, message: "API base is required" }]}
            >
              <Input placeholder="https://your-headroom-endpoint" />
            </Form.Item>
            <Form.Item name="defaultOn" label="Apply to all requests" valuePropName="checked">
              <Switch />
            </Form.Item>
            <div className="flex justify-end">
              <Button type="primary" htmlType="submit" loading={isSaving}>
                Add guardrail
              </Button>
            </div>
          </Form>
        </CardContent>
      </Card>
    </div>
  );
};

export default PromptCompressionTab;
