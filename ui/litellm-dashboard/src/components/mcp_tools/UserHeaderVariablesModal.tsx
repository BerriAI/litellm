import React, { useEffect, useState } from "react";
import { Modal, Form, Input, Typography, Alert } from "antd";
import { Button } from "@tremor/react";
import {
  getMCPUserHeaderVariables,
  putMCPUserHeaderVariables,
  MCPUserHeaderVariablesStatus,
} from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import { MCPServer } from "./types";

interface UserHeaderVariablesModalProps {
  server: MCPServer;
  open: boolean;
  accessToken: string;
  onClose: () => void;
  /** Called after a successful save with the new status. */
  onSaved?: (status: MCPUserHeaderVariablesStatus) => void;
}

/**
 * Modal for users to fill in the per-user header-variable values for a single
 * MCP server. After saving, the server is no longer flagged as "setup required"
 * on the dashboard for this user.
 */
export const UserHeaderVariablesModal: React.FC<UserHeaderVariablesModalProps> = ({
  server,
  open,
  accessToken,
  onClose,
  onSaved,
}) => {
  const [form] = Form.useForm();
  const [status, setStatus] = useState<MCPUserHeaderVariablesStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [loadError, setLoadError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) {
      setStatus(null);
      setLoadError(null);
      return;
    }
    let cancelled = false;
    setIsLoading(true);
    setLoadError(null);
    getMCPUserHeaderVariables(accessToken, server.server_id)
      .then((s) => {
        if (cancelled) return;
        setStatus(s);
        const initial: Record<string, string> = {};
        s.user_variables.forEach((name) => {
          initial[name] = s.values?.[name] ?? "";
        });
        form.setFieldsValue(initial);
      })
      .catch((err) => {
        if (cancelled) return;
        setLoadError(err instanceof Error ? err.message : String(err));
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [open, server.server_id, accessToken, form]);

  const handleSave = async (values: Record<string, string>) => {
    setIsSaving(true);
    try {
      // Drop empty strings so unsetting a variable is supported (saves it as
      // empty rather than as the literal whitespace).
      const cleaned: Record<string, string> = {};
      for (const name of status?.user_variables ?? []) {
        const v = values[name];
        cleaned[name] = typeof v === "string" ? v.trim() : "";
      }
      const next = await putMCPUserHeaderVariables(accessToken, server.server_id, cleaned);
      NotificationsManager.success("Saved your MCP credentials");
      onSaved?.(next);
      onClose();
    } catch (err) {
      NotificationsManager.fromBackend(
        err instanceof Error ? err.message : "Failed to save credentials",
      );
    } finally {
      setIsSaving(false);
    }
  };

  const displayName = server.alias || server.server_name || server.server_id;

  return (
    <Modal
      open={open}
      title={
        <div>
          <Typography.Title level={5} className="!mb-1">
            Your credentials for {displayName}
          </Typography.Title>
          <Typography.Text type="secondary" className="text-xs">
            These values are saved per-user and used only when you call this MCP server.
          </Typography.Text>
        </div>
      }
      onCancel={onClose}
      footer={null}
      width={560}
      destroyOnClose
    >
      {loadError && (
        <Alert
          type="error"
          showIcon
          className="mb-4"
          message="Failed to load credentials"
          description={loadError}
        />
      )}
      {!loadError && !isLoading && status && status.user_variables.length === 0 && (
        <Alert
          type="info"
          showIcon
          className="mb-4"
          message="This server has no per-user header variables defined."
        />
      )}
      {isLoading && <Typography.Text>Loading…</Typography.Text>}
      {!isLoading && status && status.user_variables.length > 0 && (
        <Form form={form} layout="vertical" onFinish={handleSave}>
          {status.user_variables.map((name) => {
            const missing = (status.missing_variables ?? []).includes(name);
            return (
              <Form.Item
                key={name}
                label={
                  <span>
                    <code className="text-xs bg-gray-100 px-1 py-0.5 rounded">${`{${name}}`}</code>
                    {missing && (
                      <span className="ml-2 text-xs text-red-600 font-medium">required</span>
                    )}
                  </span>
                }
                name={name}
                rules={
                  missing
                    ? [
                        {
                          required: true,
                          message: `Please provide a value for ${name}`,
                        },
                      ]
                    : []
                }
              >
                <Input.Password
                  placeholder={`Enter ${name}`}
                  size="large"
                  className="rounded-lg"
                  visibilityToggle
                />
              </Form.Item>
            );
          })}
          <div className="flex items-center justify-end space-x-3 pt-2 border-t border-gray-100 mt-4">
            <Button variant="secondary" onClick={onClose}>
              Cancel
            </Button>
            <Button variant="primary" loading={isSaving}>
              Save credentials
            </Button>
          </div>
        </Form>
      )}
    </Modal>
  );
};

export default UserHeaderVariablesModal;
