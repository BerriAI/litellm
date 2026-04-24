"use client";

import React, { useEffect, useState } from "react";
import { Form, Input, Modal, Typography } from "antd";
import type { MemoryRow } from "../networking";

const { Text } = Typography;

interface MemoryEditModalProps {
  open: boolean;
  mode: "create" | "edit";
  initialRow?: MemoryRow;
  onClose: () => void;
  onSave: (
    key: string,
    value: string,
    metadataText: string,
    isCreate: boolean,
  ) => Promise<boolean>;
}

export const MemoryEditModal: React.FC<MemoryEditModalProps> = ({
  open,
  mode,
  initialRow,
  onClose,
  onSave,
}) => {
  const [form] = Form.useForm();
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    if (!open) return;
    if (mode === "edit" && initialRow) {
      form.setFieldsValue({
        key: initialRow.key,
        value: initialRow.value,
        metadata:
          initialRow.metadata != null
            ? JSON.stringify(initialRow.metadata, null, 2)
            : "",
      });
    } else {
      form.resetFields();
    }
  }, [open, mode, initialRow, form]);

  const handleOk = async () => {
    const values = await form.validateFields();
    setSubmitting(true);
    const ok = await onSave(
      values.key.trim(),
      values.value ?? "",
      values.metadata ?? "",
      mode === "create",
    );
    setSubmitting(false);
    if (ok) {
      form.resetFields();
      onClose();
    }
  };

  return (
    <Modal
      open={open}
      title={mode === "create" ? "Create memory" : `Edit ${initialRow?.key ?? ""}`}
      onCancel={() => {
        form.resetFields();
        onClose();
      }}
      onOk={handleOk}
      okText={mode === "create" ? "Create" : "Save"}
      confirmLoading={submitting}
      width={640}
      destroyOnClose
    >
      <Form form={form} layout="vertical">
        <Form.Item
          label="Key"
          name="key"
          rules={[{ required: true, message: "Key is required" }]}
          tooltip="Globally unique — two memories cannot share a key. Namespace your own keys if you need per-user isolation (e.g. user:123:notes)."
        >
          <Input
            placeholder="e.g. user_role"
            disabled={mode === "edit"}
          />
        </Form.Item>
        <Form.Item
          label="Value"
          name="value"
          rules={[{ required: true, message: "Value is required" }]}
          tooltip="Markdown/text injected into LLM context. Plain strings are fine."
        >
          <Input.TextArea
            rows={8}
            placeholder="What the agent should remember…"
          />
        </Form.Item>
        <Form.Item
          label={
            <span>
              Metadata <Text type="secondary">(optional JSON)</Text>
            </span>
          }
          name="metadata"
          tooltip="Optional structured metadata — must be valid JSON if provided."
        >
          <Input.TextArea
            rows={4}
            placeholder='{"tags": ["example"]}'
            style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace" }}
          />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default MemoryEditModal;
