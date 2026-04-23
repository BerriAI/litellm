import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { CheckOutlined, CopyOutlined, SyncOutlined } from "@ant-design/icons";
import { Alert, Button, Col, Flex, Form, Input, InputNumber, Modal, Row, Space, Typography } from "antd";
import { add } from "date-fns";
import { useEffect, useState } from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { KeyResponse } from "../key_team_helpers/key_list";
import NotificationManager from "../molecules/notifications_manager";
import { regenerateKeyCall } from "../networking";

const { Text } = Typography;

interface RegenerateKeyModalProps {
  selectedToken: KeyResponse | null;
  visible: boolean;
  onClose: () => void;
  onKeyUpdate?: (updatedKeyData: Partial<KeyResponse>) => void;
}

export function RegenerateKeyModal({ selectedToken, visible, onClose, onKeyUpdate }: RegenerateKeyModalProps) {
  const { accessToken } = useAuthorized();
  const [form] = Form.useForm();
  const [regeneratedKey, setRegeneratedKey] = useState<string | null>(null);
  const [regenerateFormData, setRegenerateFormData] = useState<any>(null);
  const [newExpiryTime, setNewExpiryTime] = useState<string | null>(null);
  const [isRegenerating, setIsRegenerating] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (visible && selectedToken && accessToken) {
      form.setFieldsValue({
        key_alias: selectedToken.key_alias,
        max_budget: selectedToken.max_budget,
        tpm_limit: selectedToken.tpm_limit,
        rpm_limit: selectedToken.rpm_limit,
        duration: selectedToken.duration || "",
        grace_period: "",
      });
    }
  }, [visible, selectedToken, form, accessToken]);

  const calculateNewExpiryTime = (duration: string | undefined): string | null => {
    if (!duration) return null;

    try {
      const amount = parseInt(duration);
      if (Number.isNaN(amount)) {
        throw new Error("Invalid duration format");
      }
      const now = new Date();
      // Check "mo" before "m" to avoid a false prefix match (e.g. "1mo" → minutes).
      let newExpiry: Date;
      if (duration.endsWith("mo")) {
        newExpiry = add(now, { months: amount });
      } else if (duration.endsWith("s")) {
        newExpiry = add(now, { seconds: amount });
      } else if (duration.endsWith("m")) {
        newExpiry = add(now, { minutes: amount });
      } else if (duration.endsWith("h")) {
        newExpiry = add(now, { hours: amount });
      } else if (duration.endsWith("d")) {
        newExpiry = add(now, { days: amount });
      } else if (duration.endsWith("w")) {
        newExpiry = add(now, { weeks: amount });
      } else {
        throw new Error("Invalid duration format");
      }

      return newExpiry.toLocaleString();
    } catch (error) {
      return null;
    }
  };

  useEffect(() => {
    if (regenerateFormData?.duration) {
      setNewExpiryTime(calculateNewExpiryTime(regenerateFormData.duration));
    } else {
      setNewExpiryTime(null);
    }
  }, [regenerateFormData?.duration]);

  const handleRegenerateKey = async () => {
    if (!selectedToken || !accessToken) return;

    setIsRegenerating(true);
    try {
      const formValues = await form.validateFields();

      const response = await regenerateKeyCall(
        accessToken,
        selectedToken.token || selectedToken.token_id,
        formValues,
      );
      setRegeneratedKey(response.key);
      NotificationManager.success("Virtual Key regenerated successfully");

      // Build the update payload. Spread the API response first so any new
      // fields it returns (new token, timestamps, etc.) are captured, then
      // override with the explicit form values — the user's just-submitted
      // edits must win over whatever the API echoes back.
      const updatedKeyData: Partial<KeyResponse> = {
        ...response,
        token: response.token || response.key_id || selectedToken.token,
        key_name: response.key,
        max_budget: formValues.max_budget,
        tpm_limit: formValues.tpm_limit,
        rpm_limit: formValues.rpm_limit,
        expires: formValues.duration
          ? (calculateNewExpiryTime(formValues.duration) ?? selectedToken.expires)
          : selectedToken.expires,
      };

      // Update the parent component with new key data
      if (onKeyUpdate) {
        onKeyUpdate(updatedKeyData);
      }

      setIsRegenerating(false);
    } catch (error) {
      console.error("Error regenerating key:", error);
      NotificationManager.fromBackend(error);
      setIsRegenerating(false); // Reset regenerating state on error
    }
  };

  const handleClose = () => {
    setRegeneratedKey(null);
    setIsRegenerating(false);
    setCopied(false);
    form.resetFields();
    onClose();
  };

  const handleCopyKey = () => {
    setCopied(true);
  };

  return (
    <Modal
      title="Regenerate Virtual Key"
      open={visible}
      onCancel={handleClose}
      width={520}
      maskClosable={false}
      footer={
        regeneratedKey
          ? [
              <Space key="footer-actions">
                <Button onClick={handleClose}>Close</Button>
                <CopyToClipboard text={regeneratedKey} onCopy={handleCopyKey}>
                  <Button type="primary" icon={copied ? <CheckOutlined /> : <CopyOutlined />}>
                    {copied ? "Copied" : "Copy Key"}
                  </Button>
                </CopyToClipboard>
              </Space>,
            ]
          : [
              <Space key="footer-actions">
                <Button onClick={handleClose}>Cancel</Button>
                <Button type="primary" icon={<SyncOutlined />} onClick={handleRegenerateKey} loading={isRegenerating}>
                  Regenerate
                </Button>
              </Space>,
            ]
      }
    >
      {regeneratedKey ? (
        <Flex vertical gap="middle">
          <Alert type="warning" showIcon message="Save it now, you will not see it again" />

          <Flex vertical gap={2}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              Key Alias
            </Text>
            <Text>{selectedToken?.key_alias || "No alias set"}</Text>
          </Flex>

          <Flex vertical gap={6}>
            <Text type="secondary" style={{ fontSize: 12 }}>
              Virtual Key
            </Text>
            <div
              style={{
                background: "#f5f5f5",
                border: "1px solid #e8e8e8",
                borderRadius: 6,
                padding: "14px 16px",
                fontFamily: "SFMono-Regular, Consolas, 'Liberation Mono', Menlo, monospace",
                fontSize: 16,
                wordBreak: "break-all",
                color: "#262626",
              }}
            >
              {regeneratedKey}
            </div>
          </Flex>
        </Flex>
      ) : (
        <Form
          form={form}
          layout="vertical"
          style={{ marginTop: 4 }}
          onValuesChange={(changedValues) => {
            if ("duration" in changedValues) {
              setRegenerateFormData((prev: { duration?: string }) => ({ ...prev, duration: changedValues.duration }));
            }
          }}
        >
          <Form.Item name="key_alias" label="Key Alias">
            <Input disabled />
          </Form.Item>

          <Row gutter={12}>
            <Col span={8}>
              <Form.Item name="max_budget" label="Max Budget (USD)">
                <InputNumber step={0.01} precision={2} style={{ width: "100%" }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="tpm_limit" label="TPM Limit">
                <InputNumber style={{ width: "100%" }} />
              </Form.Item>
            </Col>
            <Col span={8}>
              <Form.Item name="rpm_limit" label="RPM Limit">
                <InputNumber style={{ width: "100%" }} />
              </Form.Item>
            </Col>
          </Row>

          <Row gutter={12}>
            <Col span={12}>
              <Form.Item
                name="duration"
                label="Expire Key"
                extra={
                  <Flex vertical gap={2}>
                    <Text type="secondary" style={{ fontSize: 12 }}>
                      Current expiry:{" "}
                      {selectedToken?.expires ? new Date(selectedToken.expires).toLocaleString() : "Never"}
                    </Text>
                    {newExpiryTime && (
                      <Text type="success" style={{ fontSize: 12 }}>
                        New expiry: {newExpiryTime}
                      </Text>
                    )}
                  </Flex>
                }
              >
                <Input placeholder="e.g. 30s, 30h, 30d" />
              </Form.Item>
            </Col>
            <Col span={12}>
              <Form.Item
                name="grace_period"
                label="Grace Period"
                tooltip="Keep the old key valid for this duration after rotation. Both keys work during this period for seamless cutover. Empty = immediate revoke."
                extra={
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    Recommended: 24h to 72h for production keys
                  </Text>
                }
                rules={[
                  {
                    pattern: /^(\d+(s|m|h|d|w|mo))?$/,
                    message: "Must be a duration like 30s, 30m, 24h, 2d, 1w, or 1mo",
                  },
                ]}
              >
                <Input placeholder="e.g. 24h, 2d" />
              </Form.Item>
            </Col>
          </Row>
        </Form>
      )}
    </Modal>
  );
}
