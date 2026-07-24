"use client";

import React, { useCallback, useEffect, useMemo, useState } from "react";
import { Button, Form, Input, Popconfirm, Switch } from "antd";

import { Card, CardAction, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  createGuardrailCall,
  deleteGuardrailCall,
  getGuardrailsList,
  updateGuardrailCall,
} from "@/components/networking";
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
  applyToAll: boolean;
}

const EMPTY_FORM: CompressionFormValues = { name: "", apiBase: "", applyToAll: true };

const toFormValues = (guardrail: GuardrailListItem): CompressionFormValues => ({
  name: guardrail.guardrail_name ?? "",
  apiBase: guardrail.litellm_params?.api_base ?? "",
  applyToAll: guardrail.litellm_params?.default_on ?? false,
});

const PromptCompressionTab: React.FC<PromptCompressionTabProps> = ({ accessToken }) => {
  const [form] = Form.useForm<CompressionFormValues>();
  const [existing, setExisting] = useState<GuardrailListItem | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [isSaving, setIsSaving] = useState<boolean>(false);
  const [confirmingDisable, setConfirmingDisable] = useState<boolean>(false);

  const watched = Form.useWatch([], form);

  const loadGuardrail = useCallback(() => {
    if (!accessToken) {
      return;
    }
    getGuardrailsList(accessToken)
      .then((response) => {
        const [first = null] = compressionGuardrailsOf(response as GuardrailListResponse);
        setExisting(first);
        if (first) {
          form.setFieldsValue(toFormValues(first));
        }
      })
      .catch((error) => {
        console.error("Failed to load compression guardrail:", error);
        NotificationsManager.fromBackend("Failed to load Headroom compression guardrail");
      })
      .finally(() => setIsLoading(false));
  }, [accessToken, form]);

  useEffect(() => {
    loadGuardrail();
  }, [loadGuardrail]);

  const persisted = existing ? toFormValues(existing) : null;
  const isDirty = useMemo(() => {
    if (!persisted || !watched) {
      return false;
    }
    return (
      (watched.name ?? "") !== persisted.name ||
      (watched.apiBase ?? "") !== persisted.apiBase ||
      (watched.applyToAll ?? false) !== persisted.applyToAll
    );
  }, [persisted, watched]);

  const enableGuardrail = useCallback(async () => {
    if (!accessToken) {
      return;
    }
    const values = await form.validateFields().catch(() => null);
    if (!values) {
      NotificationsManager.fromBackend("Enter a name and Headroom API base to turn on compression");
      return;
    }
    setIsSaving(true);
    try {
      await createGuardrailCall(
        accessToken,
        buildCompressionGuardrailPayload({
          name: values.name,
          apiBase: values.apiBase,
          defaultOn: values.applyToAll ?? true,
        }),
      );
      NotificationsManager.success("Headroom compression turned on");
      loadGuardrail();
    } catch (error) {
      console.error("Failed to turn on Headroom compression:", error);
      NotificationsManager.fromBackend("Failed to turn on Headroom compression");
    } finally {
      setIsSaving(false);
    }
  }, [accessToken, form, loadGuardrail]);

  const saveChanges = useCallback(async () => {
    if (!accessToken || !existing) {
      return;
    }
    const values = await form.validateFields().catch(() => null);
    if (!values) {
      return;
    }
    setIsSaving(true);
    try {
      await updateGuardrailCall(
        accessToken,
        existing.guardrail_id,
        buildCompressionGuardrailPayload({
          name: values.name,
          apiBase: values.apiBase,
          defaultOn: values.applyToAll ?? false,
        }),
      );
      NotificationsManager.success("Headroom compression updated");
      loadGuardrail();
    } catch (error) {
      console.error("Failed to update Headroom compression:", error);
      NotificationsManager.fromBackend("Failed to update Headroom compression");
    } finally {
      setIsSaving(false);
    }
  }, [accessToken, existing, form, loadGuardrail]);

  const disableGuardrail = useCallback(async () => {
    setConfirmingDisable(false);
    if (!accessToken || !existing) {
      return;
    }
    setIsSaving(true);
    try {
      await deleteGuardrailCall(accessToken, existing.guardrail_id);
      NotificationsManager.success("Headroom compression turned off");
      setExisting(null);
    } catch (error) {
      console.error("Failed to turn off Headroom compression:", error);
      NotificationsManager.fromBackend("Failed to turn off Headroom compression");
    } finally {
      setIsSaving(false);
    }
  }, [accessToken, existing]);

  const handleEnabledChange = (next: boolean) => {
    if (next) {
      enableGuardrail();
    } else {
      setConfirmingDisable(true);
    }
  };

  return (
    <div className="w-full">
      <Card>
        <CardHeader className="border-b pb-4">
          <CardTitle>Headroom prompt compression</CardTitle>
          <CardAction>
            <div className="flex items-center gap-2">
              <span className="text-sm text-muted-foreground">{existing ? "Enabled" : "Disabled"}</span>
              <Popconfirm
                open={confirmingDisable}
                title="Turn off Headroom compression?"
                description="Requests will stop being compressed and this configuration will be removed"
                okText="Turn off"
                okButtonProps={{ danger: true, loading: isSaving }}
                cancelText="Cancel"
                onConfirm={disableGuardrail}
                onCancel={() => setConfirmingDisable(false)}
              >
                <Switch
                  aria-label="Toggle Headroom compression"
                  checked={!!existing}
                  loading={isSaving}
                  disabled={isLoading}
                  onChange={handleEnabledChange}
                />
              </Popconfirm>
            </div>
          </CardAction>
        </CardHeader>
        <CardContent className="pt-6">
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

          <Form form={form} layout="vertical" requiredMark={false} initialValues={EMPTY_FORM} disabled={isSaving}>
            <Form.Item name="name" label="Name" rules={[{ required: true, message: "Name is required" }]}>
              <Input placeholder="headroom-compression" />
            </Form.Item>
            <Form.Item
              name="apiBase"
              label="Headroom API base"
              tooltip="Base URL of your Headroom compression service (LiteLLM calls its /v1/compress endpoint)"
              extra="The URL where your Headroom compression service is hosted"
              rules={[{ required: true, message: "Headroom API base is required" }]}
            >
              <Input placeholder="https://your-headroom-endpoint" />
            </Form.Item>
            <Form.Item
              name="applyToAll"
              label="Apply to all requests"
              valuePropName="checked"
              extra="On: every request is compressed. Off: compression is available for keys or teams to opt in"
            >
              <Switch />
            </Form.Item>

            <div className="mb-4 rounded-lg border border-yellow-200 bg-yellow-50 p-3">
              <p className="text-sm text-yellow-800">
                Applying compression to all requests is available to all users. Enabling it selectively per key or team
                is a LiteLLM Enterprise feature. Get a trial key{" "}
                <a
                  href="https://www.litellm.ai/#pricing"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="underline"
                >
                  here
                </a>
              </p>
            </div>

            {existing ? (
              <div className="flex justify-end">
                <Button type="primary" onClick={saveChanges} loading={isSaving} disabled={!isDirty}>
                  Save changes
                </Button>
              </div>
            ) : (
              <p className="text-sm text-muted-foreground">
                Enter a name and Headroom API base, then switch this on to start saving on input tokens
              </p>
            )}
          </Form>
        </CardContent>
      </Card>
    </div>
  );
};

export default PromptCompressionTab;
