import React, { useState } from "react";
import { Modal, Form, Input, Button } from "antd";
import { registerClaudeCodeMarketplace } from "@/components/networking";
import NotificationsManager from "@/components/molecules/notifications_manager";

interface AddMarketplaceFormProps {
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  onSuccess: () => void;
}

interface AddMarketplaceFormValues {
  source: string;
  name?: string;
}

const AddMarketplaceForm: React.FC<AddMarketplaceFormProps> = ({ visible, onClose, accessToken, onSuccess }) => {
  const [form] = Form.useForm();
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleCancel = () => {
    form.resetFields();
    onClose();
  };

  const handleSubmit = async (values: AddMarketplaceFormValues) => {
    if (!accessToken) {
      NotificationsManager.error("No access token available");
      return;
    }

    setIsSubmitting(true);
    try {
      await registerClaudeCodeMarketplace(accessToken, {
        source: values.source.trim(),
        ...(values.name?.trim() ? { name: values.name.trim() } : {}),
      });
      NotificationsManager.success("Marketplace imported successfully");
      form.resetFields();
      onSuccess();
      onClose();
    } catch (error) {
      console.error("Error registering marketplace:", error);
      const reason = error instanceof Error && error.message ? error.message : "Failed to import marketplace";
      NotificationsManager.error(`Failed to import marketplace: ${reason}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <Modal title="Add Marketplace" open={visible} onCancel={handleCancel} footer={null} width={520} className="top-8">
      <Form form={form} layout="vertical" onFinish={handleSubmit} className="mt-4">
        <Form.Item
          label="Repository"
          name="source"
          rules={[{ required: true, message: "Please enter a repository (org/repo) or URL" }]}
          tooltip="A GitHub org/repo (e.g. anthropics/claude-code-marketplace) or a full git URL"
        >
          <Input placeholder="org/repo or https://github.com/org/repo" className="rounded-lg" />
        </Form.Item>

        <Form.Item
          label="Name (Optional)"
          name="name"
          tooltip="Marketplace identifier used to namespace its skills. Defaults to the repository name"
        >
          <Input placeholder="my-marketplace" className="rounded-lg" />
        </Form.Item>

        <Form.Item className="mb-0 mt-6">
          <div className="flex justify-end gap-2">
            <Button onClick={handleCancel} disabled={isSubmitting}>
              Cancel
            </Button>
            <Button type="primary" htmlType="submit" loading={isSubmitting}>
              {isSubmitting ? "Importing..." : "Add Marketplace"}
            </Button>
          </div>
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default AddMarketplaceForm;
