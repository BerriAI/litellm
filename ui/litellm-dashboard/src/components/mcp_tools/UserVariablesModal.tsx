import React, { useEffect, useState } from "react";
import { Modal, Form, Input, Button, Alert, Spin, Tag, Typography } from "antd";
import { MCPUserVariablesGlobalStatus } from "./types";
import {
  getMCPUserVariables,
  storeMCPUserVariables,
} from "../networking";
import NotificationsManager from "../molecules/notifications_manager";

const { Text, Title } = Typography;

interface UserVariablesModalProps {
  open: boolean;
  accessToken: string | null;
  onClose: () => void;
  onSaved?: (status: MCPUserVariablesGlobalStatus) => void;
}

/**
 * User-facing modal for filling in per-user MCP variables.
 *
 * Backed by the GLOBAL (per-user, not per-server) endpoints
 * GET / POST ``/v1/mcp/user/variables``. Each field the admin marked as
 * ``scope=user`` shows up with the admin-supplied description as the
 * placeholder. The contract is write-only: the backend never returns the
 * stored value, only whether it ``is_set``.
 */
const UserVariablesModal: React.FC<UserVariablesModalProps> = ({
  open,
  accessToken,
  onClose,
  onSaved,
}) => {
  const [form] = Form.useForm();
  const [status, setStatus] = useState<MCPUserVariablesGlobalStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  useEffect(() => {
    if (!open || !accessToken) {
      return;
    }
    let cancelled = false;
    setIsLoading(true);
    (async () => {
      try {
        const fetched = await getMCPUserVariables(accessToken);
        if (cancelled) return;
        setStatus(fetched);
        // Values are write-only — never prefill from the backend.
        form.resetFields();
      } catch (err) {
        if (!cancelled) {
          NotificationsManager.fromBackend(
            `Failed to load variables: ${err instanceof Error ? err.message : String(err)}`,
          );
        }
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [open, accessToken, form]);

  const handleSave = async (values: Record<string, string>) => {
    if (!accessToken) return;
    setIsSaving(true);
    try {
      // Only send non-empty fields (write-only contract; blank means "keep").
      const trimmed: Record<string, string> = {};
      for (const [k, v] of Object.entries(values)) {
        const value = (v ?? "").trim();
        if (value !== "") {
          trimmed[k] = value;
        }
      }
      const saved = await storeMCPUserVariables(accessToken, trimmed);
      setStatus(saved);
      NotificationsManager.success("Credentials saved");
      if (onSaved) onSaved(saved);
      onClose();
    } catch (err) {
      NotificationsManager.fromBackend(
        `Failed to save variables: ${err instanceof Error ? err.message : String(err)}`,
      );
    } finally {
      setIsSaving(false);
    }
  };

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
            These values apply to every MCP server that requires them.
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
          <Alert
            type="info"
            showIcon
            message="No per-user fields are required."
          />
        ) : (
          <>
            <Text className="text-sm text-gray-600 block">
              These values are private to you. Your admin configured these
              per-user credentials:
            </Text>
            <Form
              form={form}
              layout="vertical"
              onFinish={handleSave}
              disabled={isSaving}
            >
              {required.map((spec) => (
                <Form.Item
                  key={spec.name}
                  name={spec.name}
                  label={
                    <span className="font-mono text-sm font-semibold">
                      {spec.name}
                    </span>
                  }
                  extra={spec.description || undefined}
                  rules={
                    spec.is_set
                      ? undefined
                      : [{ required: true, message: `${spec.name} is required` }]
                  }
                >
                  <Input.Password
                    placeholder={
                      spec.is_set
                        ? "•••••• (set)"
                        : spec.description || `Enter your ${spec.name}`
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

export default UserVariablesModal;
