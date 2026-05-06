"use client";

/**
 * NewSessionDialog — collects a repo URL and provisions a new session under
 * an Agent. On success the parent navigates to the three-pane view, status
 * pill = `provisioning`.
 */
import { useState } from "react";
import { Modal, Form, Input, message } from "antd";
import { createCloudSession } from "@/lib/cloud-agents-client";
import type { CloudAgentSession } from "@/types/cloud-agents";

interface NewSessionDialogProps {
  open: boolean;
  agentId: string;
  accessToken: string | null;
  onClose: () => void;
  onCreated: (session: CloudAgentSession) => void;
}

export default function NewSessionDialog({ open, agentId, accessToken, onClose, onCreated }: NewSessionDialogProps) {
  const [form] = Form.useForm<{ repo_url: string }>();
  const [submitting, setSubmitting] = useState(false);

  const handleOk = async () => {
    try {
      const values = await form.validateFields();
      setSubmitting(true);
      const session = await createCloudSession(accessToken, {
        agent_id: agentId,
        repo_url: values.repo_url,
      });
      message.success("Session provisioning…");
      form.resetFields();
      onCreated(session);
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
      title="New session"
      open={open}
      onOk={handleOk}
      onCancel={() => {
        form.resetFields();
        onClose();
      }}
      okText="Create session"
      confirmLoading={submitting}
      destroyOnClose
      data-testid="new-session-dialog"
    >
      <Form form={form} layout="vertical" preserve={false}>
        <Form.Item
          name="repo_url"
          label="Repository URL"
          rules={[
            { required: true, message: "Repository URL is required" },
            { type: "url", message: "Must be a valid URL" },
          ]}
        >
          <Input placeholder="https://github.com/owner/repo" data-testid="new-session-repo-url" />
        </Form.Item>
      </Form>
    </Modal>
  );
}
