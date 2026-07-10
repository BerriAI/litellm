import { CopyOutlined, LockOutlined } from "@ant-design/icons";
import { Button, Card, Input, Typography } from "antd";
import { useSearchParams } from "next/navigation";
import React, { useState } from "react";
import { getSecureShareCall } from "@/components/networking";
import NotificationManager from "@/components/molecules/notifications_manager";
import { decryptSecret } from "./crypto";

const { Title, Paragraph, Text } = Typography;

interface SecureShareViewProps {
  accessToken: string | null;
}

const SecureShareView: React.FC<SecureShareViewProps> = ({ accessToken }) => {
  const searchParams = useSearchParams();
  const shareId = searchParams.get("id");

  const [password, setPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [secret, setSecret] = useState<string | null>(null);

  const handleReveal = async () => {
    if (!accessToken) {
      NotificationManager.error("You must be logged in to view a secure share.");
      return;
    }
    if (!shareId) {
      NotificationManager.error("This link is missing its share id.");
      return;
    }
    if (password.length === 0) {
      NotificationManager.error("Enter the password you were given.");
      return;
    }

    setIsLoading(true);
    try {
      const share = await getSecureShareCall(accessToken, shareId);
      try {
        const plaintext = await decryptSecret(
          { ciphertext: share.ciphertext, salt: share.salt, iv: share.iv },
          password,
        );
        setSecret(plaintext);
      } catch {
        NotificationManager.error("Wrong password, or this link has been tampered with.");
      }
    } catch (error) {
      NotificationManager.fromBackend(error);
    } finally {
      setIsLoading(false);
    }
  };

  const copySecret = async () => {
    if (secret === null) return;
    await navigator.clipboard.writeText(secret);
    NotificationManager.success("Secret copied to clipboard.");
  };

  if (!shareId) {
    return (
      <Card>
        <Title level={4}>Secure Share</Title>
        <Paragraph>This link is missing its share id.</Paragraph>
      </Card>
    );
  }

  if (secret !== null) {
    return (
      <Card>
        <Title level={4}>
          <LockOutlined /> Shared credential
        </Title>
        <Paragraph>
          <Text type="secondary">Decrypted in your browser. Copy it now; the link expires automatically.</Text>
        </Paragraph>
        <div className="flex items-center gap-2">
          <Input.TextArea readOnly rows={4} value={secret} />
          <Button icon={<CopyOutlined />} onClick={copySecret}>
            Copy
          </Button>
        </div>
      </Card>
    );
  }

  return (
    <Card>
      <Title level={4}>
        <LockOutlined /> Open secure share
      </Title>
      <Paragraph>Enter the password you received to decrypt this credential in your browser.</Paragraph>
      <div className="mb-4">
        <Input.Password
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onPressEnter={handleReveal}
          placeholder="Password"
        />
      </div>
      <Button type="primary" loading={isLoading} onClick={handleReveal}>
        Reveal secret
      </Button>
    </Card>
  );
};

export default SecureShareView;
