"use client";

import { useState } from "react";
import { Modal, Form, Input, message } from "antd";
import { createCloudAgent } from "@/lib/cloud-agents-client";

interface NewAgentDialogProps {
  open: boolean;
  onClose: () => void;
  onCreated: () => void;
  accessToken: string | null;
}

export default function NewAgentDialog({ open, onClose, onCreated, accessToken }: NewAgentDialogProps) {
  const [form] = Form.useForm<{ name: string; model: string; system_prompt?: string }>();
  const [submitting, setSubmitting] = useState(false);

  const handleOk = async () => {
    try {
      const values = await form.validateFields();
      setSubmitting(true);
      await createCloudAgent(accessToken, values);
      message.success(`Agent "${values.name}" created`);
      form.resetFields();
      onCreated();
    } catch (e) {
      if (e instanceof Error) {
        message.error(e.message);
      }
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Modal
      title="New cloud agent"
      open={open}
      onOk={handleOk}
      onCancel={() => {
        form.resetFields();
        onClose();
      }}
      okText="Create"
      confirmLoading={submitting}
      destroyOnClose
      data-testid="new-agent-dialog"
    >
      <Form form={form} layout="vertical" preserve={false}>
        <Form.Item name="name" label="Name" rules={[{ required: true, message: "Name is required" }]}>
          <Input placeholder="my-coding-agent" data-testid="new-agent-name" />
        </Form.Item>
        <Form.Item name="model" label="Model" rules={[{ required: true, message: "Model is required" }]}>
          <Input placeholder="claude-3-5-sonnet-20241022" data-testid="new-agent-model" />
        </Form.Item>
        <Form.Item name="system_prompt" label="System prompt (optional)">
          <Input.TextArea rows={3} placeholder="You are a helpful coding agent." data-testid="new-agent-prompt" />
        </Form.Item>
      </Form>
    </Modal>
  );
}
