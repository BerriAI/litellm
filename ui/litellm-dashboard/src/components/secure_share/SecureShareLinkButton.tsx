import { CopyOutlined, LockOutlined } from "@ant-design/icons";
import { Button, Form, Input, Modal, Select, Typography } from "antd";
import React, { useState } from "react";
import { createSecureShareCall, SecureShareCreatePayload } from "@/components/networking";
import NotificationManager from "@/components/molecules/notifications_manager";
import { migratedHref } from "@/utils/migratedPages";
import { encryptSecret } from "./crypto";

const { Paragraph, Text } = Typography;

const MIN_PASSWORD_LENGTH = 8;

const EXPIRY_OPTIONS = [
  { value: "1h", label: "1 hour" },
  { value: "6h", label: "6 hours" },
  { value: "1d", label: "1 day" },
  { value: "7d", label: "7 days" },
];

interface SecureShareLinkButtonProps {
  secret: string;
  accessToken: string | null;
}

interface ShareFormValues {
  password: string;
  confirmPassword: string;
  expiry: string;
}

const buildShareLink = (shareId: string): string =>
  `${window.location.origin}${migratedHref("secure-share/view")}?id=${encodeURIComponent(shareId)}`;

const SecureShareLinkButton: React.FC<SecureShareLinkButtonProps> = ({ secret, accessToken }) => {
  const [form] = Form.useForm<ShareFormValues>();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [link, setLink] = useState<string | null>(null);

  const closeModal = () => {
    setIsModalOpen(false);
    setLink(null);
    form.resetFields();
  };

  const handleGenerate = async (values: ShareFormValues) => {
    if (!accessToken) {
      NotificationManager.error("You must be logged in to create a share link.");
      return;
    }
    setIsGenerating(true);
    try {
      const encrypted = await encryptSecret(secret, values.password);
      const payload: SecureShareCreatePayload = { ...encrypted, expiry: values.expiry };
      const response = await createSecureShareCall(accessToken, payload);
      setLink(buildShareLink(response.share_id));
    } catch (error) {
      NotificationManager.fromBackend(error);
    } finally {
      setIsGenerating(false);
    }
  };

  const copyLink = async () => {
    if (link === null) return;
    await navigator.clipboard.writeText(link);
    NotificationManager.success("Share link copied to clipboard.");
  };

  return (
    <>
      <Button icon={<LockOutlined />} onClick={() => setIsModalOpen(true)} style={{ marginTop: 12, marginLeft: 8 }}>
        Generate one-time share link
      </Button>
      <Modal
        title="Share this key over a one-time encrypted link"
        open={isModalOpen}
        onCancel={closeModal}
        footer={null}
        destroyOnClose
      >
        {link === null ? (
          <>
            <Paragraph>
              <Text type="secondary">
                The key is encrypted in your browser with the password below and never reaches the server in plaintext.
                Share the link and password separately; the link opens once, then it is gone.
              </Text>
            </Paragraph>
            <Form form={form} layout="vertical" onFinish={handleGenerate} initialValues={{ expiry: "1d" }}>
              <Form.Item
                label="Password"
                name="password"
                rules={[{ required: true, min: MIN_PASSWORD_LENGTH, message: "Use at least 8 characters." }]}
              >
                <Input.Password placeholder="Password the recipient will enter" />
              </Form.Item>
              <Form.Item
                label="Confirm password"
                name="confirmPassword"
                dependencies={["password"]}
                rules={[
                  { required: true, message: "Confirm the password." },
                  ({ getFieldValue }) => ({
                    validator: (_, value) =>
                      !value || getFieldValue("password") === value
                        ? Promise.resolve()
                        : Promise.reject(new Error("Passwords do not match.")),
                  }),
                ]}
              >
                <Input.Password placeholder="Re-enter the password" />
              </Form.Item>
              <Form.Item label="Link expires after" name="expiry" rules={[{ required: true }]}>
                <Select options={EXPIRY_OPTIONS} />
              </Form.Item>
              <Button type="primary" htmlType="submit" loading={isGenerating}>
                Generate link
              </Button>
            </Form>
          </>
        ) : (
          <>
            <Paragraph>
              <Text type="secondary">
                Send this link to the recipient and give them the password out-of-band. It reveals the key once, then
                expires.
              </Text>
            </Paragraph>
            <div className="flex items-center gap-2">
              <Input readOnly value={link} />
              <Button icon={<CopyOutlined />} onClick={copyLink}>
                Copy
              </Button>
            </div>
          </>
        )}
      </Modal>
    </>
  );
};

export default SecureShareLinkButton;
