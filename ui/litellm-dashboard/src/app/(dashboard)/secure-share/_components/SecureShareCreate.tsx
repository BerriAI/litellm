import { CopyOutlined, LockOutlined } from "@ant-design/icons";
import { Button, Card, Input, Select, Typography } from "antd";
import React, { useState } from "react";
import { createSecureShareCall } from "@/components/networking";
import NotificationManager from "@/components/molecules/notifications_manager";
import { encryptSecret } from "./crypto";

const { Title, Text, Paragraph } = Typography;

const EXPIRY_OPTIONS = [
  { value: "1h", label: "1 hour" },
  { value: "6h", label: "6 hours" },
  { value: "1d", label: "1 day" },
  { value: "7d", label: "7 days" },
];

const MIN_PASSWORD_LENGTH = 8;

interface SecureShareCreateProps {
  accessToken: string | null;
}

interface CreatedShare {
  link: string;
  expiresAt: string;
}

function buildShareLink(shareId: string): string {
  const base = window.location.href.split(/[?#]/)[0].replace(/\/$/, "");
  return `${base}/view?id=${encodeURIComponent(shareId)}`;
}

const SecureShareCreate: React.FC<SecureShareCreateProps> = ({ accessToken }) => {
  const [secret, setSecret] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [expiry, setExpiry] = useState("1d");
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [created, setCreated] = useState<CreatedShare | null>(null);

  const reset = () => {
    setSecret("");
    setPassword("");
    setConfirmPassword("");
    setExpiry("1d");
    setCreated(null);
  };

  const handleCreate = async () => {
    if (!accessToken) {
      NotificationManager.error("You must be logged in to create a secure share.");
      return;
    }
    if (secret.trim().length === 0) {
      NotificationManager.error("Enter the credential you want to share.");
      return;
    }
    if (password.length < MIN_PASSWORD_LENGTH) {
      NotificationManager.error(`Password must be at least ${MIN_PASSWORD_LENGTH} characters.`);
      return;
    }
    if (password !== confirmPassword) {
      NotificationManager.error("Passwords do not match.");
      return;
    }

    setIsSubmitting(true);
    try {
      const encrypted = await encryptSecret(secret, password);
      const payload = { ...encrypted, expiry };
      const response = await createSecureShareCall(accessToken, payload);
      setCreated({ link: buildShareLink(response.share_id), expiresAt: response.expires_at });
      NotificationManager.success("Secure share created. The secret was encrypted in your browser.");
    } catch (error) {
      NotificationManager.fromBackend(error);
    } finally {
      setIsSubmitting(false);
    }
  };

  const copyLink = async () => {
    if (!created) return;
    await navigator.clipboard.writeText(created.link);
    NotificationManager.success("Link copied to clipboard.");
  };

  if (created) {
    return (
      <Card>
        <Title level={4}>
          <LockOutlined /> Secure share created
        </Title>
        <Paragraph>
          Send this link to the recipient. Share the password separately (not in the same channel). The recipient must
          be logged in as a proxy admin or internal user to open it.
        </Paragraph>
        <div className="flex items-center gap-2">
          <Input readOnly value={created.link} />
          <Button icon={<CopyOutlined />} onClick={copyLink}>
            Copy
          </Button>
        </div>
        <Paragraph className="mt-4">
          <Text type="secondary">Expires at {new Date(created.expiresAt).toLocaleString()}</Text>
        </Paragraph>
        <Button type="primary" onClick={reset}>
          Share another
        </Button>
      </Card>
    );
  }

  return (
    <Card>
      <Title level={4}>
        <LockOutlined /> Secure Share
      </Title>
      <Paragraph>
        Share a credential over a temporary, end-to-end encrypted link. The secret is encrypted in your browser with a
        key derived from the password you choose; the server only ever stores ciphertext.
      </Paragraph>

      <div className="mb-4">
        <Text>Credential</Text>
        <Input.TextArea
          rows={4}
          value={secret}
          onChange={(e) => setSecret(e.target.value)}
          placeholder="Paste the API key or secret to share"
        />
      </div>

      <div className="mb-4">
        <Text>Password</Text>
        <Input.Password
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder={`At least ${MIN_PASSWORD_LENGTH} characters`}
        />
      </div>

      <div className="mb-4">
        <Text>Confirm password</Text>
        <Input.Password
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
          placeholder="Re-enter the password"
        />
      </div>

      <div className="mb-4">
        <Text>Expires after</Text>
        <Select value={expiry} onChange={setExpiry} options={EXPIRY_OPTIONS} className="w-full" />
      </div>

      <Button type="primary" loading={isSubmitting} onClick={handleCreate}>
        Create secure link
      </Button>
    </Card>
  );
};

export default SecureShareCreate;
