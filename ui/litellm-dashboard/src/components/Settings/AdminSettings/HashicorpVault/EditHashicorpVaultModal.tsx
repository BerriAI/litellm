"use client";

import { useHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useHashicorpVaultConfig";
import { useUpdateHashicorpVaultConfig } from "@/app/(dashboard)/hooks/configOverrides/useUpdateHashicorpVaultConfig";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import NotificationManager from "@/components/molecules/notifications_manager";
import { Button, Divider, Form, Input, Modal, Space, Typography } from "antd";
import React, { useEffect, useMemo } from "react";
import { useTranslation } from "react-i18next";
import type { TFunction } from "i18next";
import { SENSITIVE_FIELDS, FIELD_LABELS } from "./constants";

type TranslateFn = TFunction;

interface FieldGroup {
  title: string;
  subtitle?: string;
  fields: string[];
}

const getFieldGroups = (t: TFunction): FieldGroup[] => [
  {
    title: t("settingsPages.editHashicorpVaultModal.groupConnection"),
    fields: ["vault_addr", "vault_namespace", "vault_mount_name", "vault_path_prefix"],
  },
  {
    title: t("settingsPages.editHashicorpVaultModal.groupTokenAuth"),
    subtitle: t("settingsPages.editHashicorpVaultModal.groupTokenAuthSubtitle"),
    fields: ["vault_token"],
  },
  {
    title: t("settingsPages.editHashicorpVaultModal.groupAppRoleAuth"),
    subtitle: t("settingsPages.editHashicorpVaultModal.groupAppRoleAuthSubtitle"),
    fields: ["approle_role_id", "approle_secret_id", "approle_mount_path"],
  },
  {
    title: t("settingsPages.editHashicorpVaultModal.groupTls"),
    subtitle: t("settingsPages.editHashicorpVaultModal.groupTlsSubtitle"),
    fields: ["client_cert", "client_key", "vault_cert_role"],
  },
];

interface EditHashicorpVaultModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess: () => void;
}

const EditHashicorpVaultModal: React.FC<EditHashicorpVaultModalProps> = ({ isVisible, onCancel, onSuccess }) => {
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const { accessToken } = useAuthorized();
  const { data } = useHashicorpVaultConfig();
  const { mutate, isPending } = useUpdateHashicorpVaultConfig(accessToken);

  const fieldGroups = useMemo(() => getFieldGroups(t), [t]);

  const schema = data?.field_schema;
  const properties = schema?.properties ?? {};
  const rawValues = data?.values ?? {};

  useEffect(() => {
    if (isVisible && data) {
      form.resetFields();
      // Only set non-sensitive fields — sensitive ones show as placeholders
      const formValues: Record<string, any> = {};
      for (const [key, value] of Object.entries(rawValues)) {
        if (!SENSITIVE_FIELDS.has(key)) {
          formValues[key] = value;
        }
      }
      form.setFieldsValue(formValues);
    }
  }, [isVisible, data, form]);

  const handleSubmit = (formValues: Record<string, any>) => {
    const config: Record<string, any> = {};
    for (const [key, value] of Object.entries(formValues)) {
      if (value !== undefined && value !== null && value !== "") {
        // Non-empty value → update
        config[key] = value;
      } else if (!SENSITIVE_FIELDS.has(key)) {
        // Non-sensitive field cleared → send "" to clear it on the backend
        config[key] = "";
      }
      // Sensitive field left blank → omit from payload (keep existing)
    }

    mutate(config, {
      onSuccess: () => {
        NotificationManager.success(t("settingsPages.editHashicorpVaultModal.saveSuccess"));
        onSuccess();
      },
      onError: (err) => {
        NotificationManager.fromBackend(err);
      },
    });
  };

  const handleCancel = () => {
    form.resetFields();
    onCancel();
  };

  const renderField = (fieldName: string) => {
    const fieldSchema = properties[fieldName];
    if (!fieldSchema) return null;

    const rules =
      fieldName === "vault_addr"
        ? [{ pattern: /^https?:\/\/.+/, message: t("settingsPages.editHashicorpVaultModal.urlPatternMessage") }]
        : undefined;

    const isSensitive = SENSITIVE_FIELDS.has(fieldName);
    const existingValue = rawValues[fieldName];
    const hasExistingValue = isSensitive && existingValue != null && existingValue !== "";
    const placeholder = hasExistingValue
      ? t("settingsPages.editHashicorpVaultModal.leaveBlankToKeep", { existing: existingValue })
      : fieldSchema?.description;

    return (
      <Form.Item key={fieldName} name={fieldName} label={FIELD_LABELS[fieldName] ?? fieldName} rules={rules}>
        {isSensitive ? <Input.Password placeholder={placeholder} /> : <Input placeholder={fieldSchema?.description} />}
      </Form.Item>
    );
  };

  return (
    <Modal
      title={t("settingsPages.editHashicorpVaultModal.title")}
      open={isVisible}
      width={700}
      footer={
        <Space>
          <Button onClick={handleCancel} disabled={isPending}>
            {t("common.cancel")}
          </Button>
          <Button type="primary" loading={isPending} onClick={() => form.submit()}>
            {isPending ? t("common.saving") : t("common.save")}
          </Button>
        </Space>
      }
      onCancel={handleCancel}
    >
      <Form form={form} layout="vertical" onFinish={handleSubmit}>
        {fieldGroups.map((group, index) => (
          <div key={group.title}>
            {index > 0 && <Divider />}
            <Typography.Title level={5} style={{ marginBottom: 4 }}>
              {group.title}
            </Typography.Title>
            {group.subtitle && (
              <Typography.Paragraph type="secondary" style={{ marginBottom: 16 }}>
                {group.subtitle}
              </Typography.Paragraph>
            )}
            {group.fields.map(renderField)}
          </div>
        ))}
      </Form>
    </Modal>
  );
};

export default EditHashicorpVaultModal;
