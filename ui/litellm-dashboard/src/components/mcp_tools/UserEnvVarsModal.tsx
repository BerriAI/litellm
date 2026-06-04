import React, { useEffect, useState } from "react";
import { Modal, Form, Input, Button, Alert, Spin, Tag, Typography } from "antd";
import { MCPServer, MCPUserEnvVarsStatus } from "./types";
import { getMCPUserEnvVars, storeMCPUserEnvVars } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

const { Text, Title } = Typography;

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
const UserEnvVarsModal: React.FC<UserEnvVarsModalProps> = ({ server, open, accessToken, onClose, onSaved }) => {
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
        form.resetFields();
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
      NotificationsManager.success("Credentials saved");
      if (onSaved) onSaved(saved);
      onClose();
    } catch (err) {
      NotificationsManager.fromBackend(`Failed to save env vars: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setIsSaving(false);
    }
  };

  const displayName = server?.server_name || server?.alias || server?.server_id || "MCP Server";
  const required = status?.required ?? [];

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={520}
      destroyOnHidden
      title={
        <div>
          <div className="flex items-center gap-2">
            <Title level={5} style={{ margin: 0 }}>
              Set your credentials
            </Title>
            <Tag color="blue">Per-user</Tag>
          </div>
          <Text type="secondary" className="text-xs">
            {displayName}
          </Text>
        </div>
      }
    >
      <div className="space-y-4 mt-2">
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Spin />
          </div>
        ) : required.length === 0 ? (
          <Alert type="info" showIcon message="No per-user fields configured for this server." />
        ) : (
          <>
            <Text className="text-sm text-gray-600 block">
              These values are private to you. Your admin configured this MCP server to require these per-user
              credentials. Saved values are never shown back; leave an already-set field blank to keep it, or enter a
              value to set or change it.
            </Text>
            <Form form={form} layout="vertical" onFinish={handleSave} disabled={isSaving}>
              {required.map((spec) => (
                <Form.Item
                  key={spec.name}
                  name={spec.name}
                  label={
                    <span className="flex items-center gap-2">
                      <span className="font-mono text-sm font-semibold">{spec.name}</span>
                      {spec.is_set && <Tag color="green">Set</Tag>}
                    </span>
                  }
                  extra={spec.description || undefined}
                  rules={spec.is_set ? undefined : [{ required: true, message: `${spec.name} is required` }]}
                >
                  <Input.Password
                    placeholder={
                      spec.is_set ? "Enter a new value to overwrite" : spec.description || `Enter your ${spec.name}`
                    }
                    visibilityToggle
                  />
                </Form.Item>
              ))}
              <div className="flex items-center justify-end gap-2 pt-2 border-t border-gray-100">
                <Button onClick={onClose} disabled={isSaving}>
                  Cancel
                </Button>
                <Button type="primary" htmlType="submit" loading={isSaving}>
                  Save Credentials
                </Button>
              </div>
            </Form>
          </>
        )}
      </div>
    </Modal>
  );
};

export default UserEnvVarsModal;
