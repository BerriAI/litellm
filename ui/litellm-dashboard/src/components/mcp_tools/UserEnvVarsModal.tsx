import React, { useEffect, useState } from "react";
import { Modal, Form, Input, Button, Alert, Typography } from "antd";
import { MCPServer, MCPUserEnvVarsStatus } from "./types";
import {
  getMCPUserEnvVars,
  storeMCPUserEnvVars,
} from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

const { Text, Title, Paragraph } = Typography;

interface UserEnvVarsModalProps {
  server: MCPServer | null;
  open: boolean;
  accessToken: string | null;
  onClose: () => void;
  onSaved?: (status: MCPUserEnvVarsStatus) => void;
}

/**
 * User-facing modal for filling in per-user MCP environment variables.
 *
 * Backed by GET / POST ``/v1/mcp/server/{id}/user-env-vars``. Each field
 * the admin marked as ``scope=user`` shows up with the admin-supplied
 * description as the placeholder.
 */
const UserEnvVarsModal: React.FC<UserEnvVarsModalProps> = ({
  server,
  open,
  accessToken,
  onClose,
  onSaved,
}) => {
  const [form] = Form.useForm();
  const [status, setStatus] = useState<MCPUserEnvVarsStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (!open || !server || !accessToken) {
      return;
    }
    let cancelled = false;
    setIsLoading(true);
    (async () => {
      try {
        const fetched = await getMCPUserEnvVars(accessToken, server.server_id);
        if (cancelled) return;
        setStatus(fetched);
        const initial: Record<string, string> = {};
        for (const spec of fetched?.required ?? []) {
          initial[spec.name] = spec.value ?? "";
        }
        form.setFieldsValue(initial);
      } catch (err) {
        if (!cancelled) {
          NotificationsManager.fromBackend(
            `Failed to load env vars: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, server, accessToken, form]);

  const handleSave = async (values: Record<string, string>) => {
    if (!server || !accessToken) return;
    setIsSaving(true);
    try {
      const trimmed: Record<string, string> = {};
      for (const [k, v] of Object.entries(values)) {
        trimmed[k] = (v ?? "").trim();
      }
      const saved = await storeMCPUserEnvVars(accessToken, server.server_id, trimmed);
      setStatus(saved);
      NotificationsManager.success("Environment variables saved");
      if (onSaved) onSaved(saved);
      onClose();
    } catch (err) {
      NotificationsManager.fromBackend(
        `Failed to save env vars: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setIsSaving(false);
    }
  };

  const displayName = server?.alias || server?.server_name || server?.server_id || "MCP Server";

  return (
    <Modal
      title={
        <div>
          <Title level={4} className="!mb-0">
            Set your credentials for {displayName}
          </Title>
          <Text type="secondary" className="text-sm">
            These values are stored only for you and are injected into the MCP server&apos;s
            request headers when you use it.
          </Text>
        </div>
      }
      open={open}
      onCancel={onClose}
      footer={null}
      width={580}
      destroyOnHidden
    >
      {status && status.required.length === 0 ? (
        <Alert
          type="info"
          showIcon
          message="This MCP server doesn't require any per-user values."
        />
      ) : (
        <Form
          form={form}
          layout="vertical"
          onFinish={handleSave}
          disabled={isLoading || isSaving}
        >
          {status?.missing_count ? (
            <Alert
              type="warning"
              showIcon
              className="mb-4"
              message={`${status.missing_count} required field${status.missing_count === 1 ? "" : "s"} missing`}
            />
          ) : null}

          {(status?.required ?? []).map((spec) => (
            <Form.Item
              key={spec.name}
              name={spec.name}
              label={<span className="font-mono text-sm">{spec.name}</span>}
              extra={spec.description || undefined}
              rules={[{ required: true, message: `${spec.name} is required` }]}
            >
              <Input
                size="large"
                placeholder={spec.description || `Enter value for ${spec.name}`}
                allowClear
              />
            </Form.Item>
          ))}

          {(status?.required ?? []).length === 0 && !isLoading && (
            <Paragraph type="secondary">
              No per-user variables required for this server.
            </Paragraph>
          )}

          <div className="flex justify-end gap-2 mt-4">
            <Button onClick={onClose} disabled={isSaving}>
              Cancel
            </Button>
            <Button
              type="primary"
              htmlType="submit"
              loading={isSaving}
              disabled={(status?.required ?? []).length === 0}
            >
              Save
            </Button>
          </div>
        </Form>
      )}
    </Modal>
  );
};

export default UserEnvVarsModal;
