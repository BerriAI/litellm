"use client";

import React, { useEffect, useState } from "react";
import { Form, Input, Modal, Typography } from "antd";
import { Trans, useTranslation } from "react-i18next";
import type { MemoryRow } from "@/components/networking";

const { Text } = Typography;

interface MemoryEditModalProps {
  open: boolean;
  mode: "create" | "edit";
  initialRow?: MemoryRow;
  onClose: () => void;
  onSave: (key: string, value: string, metadataText: string, isCreate: boolean) => Promise<boolean>;
}

export const MemoryEditModal: React.FC<MemoryEditModalProps> = ({ open, mode, initialRow, onClose, onSave }) => {
  const { t } = useTranslation();
  const [form] = Form.useForm();
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!open) return;
    if (mode === "edit" && initialRow) {
      form.setFieldsValue({
        key: initialRow.key,
        value: initialRow.value,
        metadata: initialRow.metadata != null ? JSON.stringify(initialRow.metadata, null, 2) : "",
      });
    } else {
      form.resetFields();
    }
  }, [open, mode, initialRow, form]);

  const handleOk = async () => {
    const values = await form.validateFields();
    setSubmitting(true);
    const ok = await onSave(values.key.trim(), values.value ?? "", values.metadata ?? "", mode === "create");
    setSubmitting(false);
    if (ok) {
      form.resetFields();
      onClose();
    }
  };

  return (
    <Modal
      open={open}
      title={
        mode === "create"
          ? t("memoryView.memoryEditModal.titleCreate")
          : t("memoryView.memoryEditModal.titleEdit", { key: initialRow?.key ?? "" })
      }
      onCancel={() => {
        form.resetFields();
        onClose();
      }}
      onOk={handleOk}
      okText={mode === "create" ? t("common.create") : t("common.save")}
      confirmLoading={submitting}
      width={640}
      destroyOnClose
    >
      <Form form={form} layout="vertical">
        <Form.Item
          label={t("memoryView.memoryEditModal.labelKey")}
          name="key"
          rules={[{ required: true, message: t("memoryView.memoryEditModal.keyRequired") }]}
          tooltip={t("memoryView.memoryEditModal.keyTooltip")}
        >
          <Input placeholder={t("memoryView.memoryEditModal.keyPlaceholder")} disabled={mode === "edit"} />
        </Form.Item>
        <Form.Item
          label={t("memoryView.memoryEditModal.labelValue")}
          name="value"
          rules={[{ required: true, message: t("memoryView.memoryEditModal.valueRequired") }]}
          tooltip={t("memoryView.memoryEditModal.valueTooltip")}
        >
          <Input.TextArea rows={8} placeholder={t("memoryView.memoryEditModal.valuePlaceholder")} />
        </Form.Item>
        <Form.Item
          label={
            <Trans
              i18nKey="memoryView.memoryEditModal.labelMetadata"
              components={{ secondary: <Text type="secondary" /> }}
            />
          }
          name="metadata"
          tooltip={t("memoryView.memoryEditModal.metadataTooltip")}
        >
          <Input.TextArea
            rows={4}
            placeholder={t("memoryView.memoryEditModal.metadataPlaceholder")}
            style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}
          />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default MemoryEditModal;
