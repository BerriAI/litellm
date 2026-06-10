import React from "react";
import { Modal, Form, Input, Button, Alert, Spin, Tag, Typography } from "antd";
import { useMutation, useQuery } from "@tanstack/react-query";
import { MCPServer, MCPUserEnvVarsStatus } from "./types";
import { getMCPUserEnvVars, storeMCPUserEnvVars } from "../networking";
import NotificationsManager from "../molecules/notifications_manager";
import { useTranslation } from "react-i18next";

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
  const { t } = useTranslation();

  const {
    data: status,
    isLoading,
    isError,
  } = useQuery<MCPUserEnvVarsStatus>({
    queryKey: ["mcpUserEnvVars", server?.server_id],
    queryFn: () => getMCPUserEnvVars(accessToken!, server!.server_id),
    enabled: open && !!server && !!accessToken,
  });

  const saveMutation = useMutation({
    mutationFn: (values: Record<string, string>) => storeMCPUserEnvVars(accessToken!, server!.server_id, values),
    onSuccess: (saved) => {
      NotificationsManager.success(t("mcpTools.userEnvVarsModal.savedSuccess"));
      onSaved?.(saved);
      onClose();
    },
    onError: (err) => {
      NotificationsManager.fromBackend(
        t("mcpTools.userEnvVarsModal.saveFailed", {
          error: err instanceof Error ? err.message : String(err),
        }),
      );
    },
  });

  const handleSave = (values: Record<string, string>) => {
    if (!server || !accessToken) return;
    const trimmed: Record<string, string> = {};
    for (const [k, v] of Object.entries(values)) {
      trimmed[k] = (v ?? "").trim();
    }
    saveMutation.mutate(trimmed);
  };

  const displayName = server?.server_name || server?.alias || server?.server_id || "MCP Server";
  const required = status?.required ?? [];
  const isSaving = saveMutation.isPending;

  return (
    <Modal
      open={open}
      onCancel={onClose}
      footer={null}
      width={520}
      destroyOnHidden
      afterOpenChange={(opened) => {
        if (opened) form.resetFields();
      }}
      title={
        <div>
          <div className="flex items-center gap-2">
            <Title level={5} style={{ margin: 0 }}>
              {t("mcpTools.userEnvVarsModal.title")}
            </Title>
            <Tag color="blue">{t("mcpTools.userEnvVarsModal.perUserTag")}</Tag>
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
        ) : isError ? (
          <Alert type="error" showIcon message={t("mcpTools.userEnvVarsModal.loadError")} />
        ) : required.length === 0 ? (
          <Alert type="info" showIcon message={t("mcpTools.userEnvVarsModal.noFieldsConfigured")} />
        ) : (
          <>
            <Text className="text-sm text-gray-600 block">{t("mcpTools.userEnvVarsModal.privacyNote")}</Text>
            <Form form={form} layout="vertical" onFinish={handleSave} disabled={isSaving}>
              {required.map((spec) => (
                <Form.Item
                  key={spec.name}
                  name={spec.name}
                  label={
                    <span className="flex items-center gap-2">
                      <span className="font-mono text-sm font-semibold">{spec.name}</span>
                      {spec.is_set && <Tag color="green">{t("mcpTools.userEnvVarsModal.setTag")}</Tag>}
                    </span>
                  }
                  extra={spec.description || undefined}
                  rules={
                    spec.is_set
                      ? undefined
                      : [{ required: true, message: t("mcpTools.userEnvVarsModal.fieldRequired", { name: spec.name }) }]
                  }
                >
                  <Input.Password
                    placeholder={
                      spec.is_set
                        ? t("mcpTools.userEnvVarsModal.overwritePlaceholder")
                        : spec.description || t("mcpTools.userEnvVarsModal.enterFieldPlaceholder", { name: spec.name })
                    }
                    visibilityToggle
                  />
                </Form.Item>
              ))}
              <div className="flex items-center justify-end gap-2 pt-2 border-t border-gray-100">
                <Button onClick={onClose} disabled={isSaving}>
                  {t("common.cancel")}
                </Button>
                <Button type="primary" htmlType="submit" loading={isSaving}>
                  {t("mcpTools.userEnvVarsModal.saveButton")}
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
