import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Button, Form, Input, InputNumber, Modal, Typography } from "antd";
import { add } from "date-fns";
import { useEffect, useState } from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { KeyResponse } from "../key_team_helpers/key_list";
import NotificationManager from "../molecules/notifications_manager";
import { regenerateKeyCall } from "../networking";

const { Text, Title } = Typography;

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

  // Track whether this is the user's own authentication key
  const [isOwnKey, setIsOwnKey] = useState<boolean>(false);

  // Keep track of the current valid access token locally
  const [currentAccessToken, setCurrentAccessToken] = useState<string | null>(null);

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

      // Initialize the current access token
      setCurrentAccessToken(accessToken);

      // Check if this is the user's own authentication key by comparing the key values
      const isUserOwnKey = selectedToken.key_name === accessToken;
      setIsOwnKey(isUserOwnKey);
    }
  }, [visible, selectedToken, form, accessToken]);

  useEffect(() => {
    if (!visible) {
      // Reset states when modal is closed
      setRegeneratedKey(null);
      setIsRegenerating(false);
      setIsOwnKey(false);
      setCurrentAccessToken(null);
      form.resetFields();
    }
  }, [visible, form]);

  const calculateNewExpiryTime = (duration: string | undefined): string | null => {
    if (!duration) return null;

    try {
      const now = new Date();
      let newExpiry: Date;

      if (duration.endsWith("s")) {
        newExpiry = add(now, { seconds: parseInt(duration) });
      } else if (duration.endsWith("h")) {
        newExpiry = add(now, { hours: parseInt(duration) });
      } else if (duration.endsWith("d")) {
        newExpiry = add(now, { days: parseInt(duration) });
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
    if (!selectedToken || !currentAccessToken) return;

    setIsRegenerating(true);
    try {
      const formValues = await form.validateFields();

      // Use the current access token for the API call
      const response = await regenerateKeyCall(
        currentAccessToken,
        selectedToken.token || selectedToken.token_id,
        formValues,
      );
      setRegeneratedKey(response.key);
      NotificationManager.success("Virtual Key regenerated successfully");

      console.log("Full regenerate response:", response); // Debug log to see what's returned

      // Create updated key data with ALL new values from the response
      const updatedKeyData: Partial<KeyResponse> = {
        // Use the new token/key ID from the response (this is what was missing!)
        token: response.token || response.key_id || selectedToken.token, // Try different possible field names
        key_name: response.key, // This is the new secret key string
        max_budget: formValues.max_budget,
        tpm_limit: formValues.tpm_limit,
        rpm_limit: formValues.rpm_limit,
        expires: formValues.duration ? calculateNewExpiryTime(formValues.duration) : selectedToken.expires,
        // Include any other fields that might be returned by the API
        ...response, // Spread the entire response to capture all updated fields
      };

      console.log("Updated key data with new token:", updatedKeyData); // Debug log

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
    setIsOwnKey(false);
    setCurrentAccessToken(null);
    form.resetFields();
    onClose();
  };

  return (
    <Modal
      title="Regenerate Virtual Key"
      open={visible}
      onCancel={handleClose}
      footer={
        regeneratedKey
          ? [
              <Button key="close" onClick={handleClose}>
                Close
              </Button>,
            ]
          : [
              <Button key="cancel" onClick={handleClose} style={{ marginRight: 8 }}>
                Cancel
              </Button>,
              <Button key="regenerate" type="primary" onClick={handleRegenerateKey} loading={isRegenerating}>
                Regenerate
              </Button>,
            ]
      }
    >
      {regeneratedKey ? (
        <div>
          <Title level={4}>Regenerated Key</Title>
          <p>
            Please replace your old key with the new key generated. For security reasons,{" "}
            <b>you will not be able to view it again</b> through your LiteLLM account. If you lose this secret key,
            you will need to generate a new one.
          </p>
          <Text type="secondary" style={{ marginTop: 12, display: "block" }}>Key Alias:</Text>
          <div style={{ background: "#f8f8f8", padding: 8, borderRadius: 4, marginBottom: 8 }}>
            <pre style={{ wordWrap: "break-word", whiteSpace: "normal", margin: 0 }}>
              {selectedToken?.key_alias || "No alias set"}
            </pre>
          </div>
          <Text type="secondary" style={{ marginTop: 12, display: "block" }}>New Virtual Key:</Text>
          <div style={{ background: "#f8f8f8", padding: 8, borderRadius: 4, marginBottom: 8 }}>
            <pre style={{ wordWrap: "break-word", whiteSpace: "normal", margin: 0 }}>
              {regeneratedKey}
            </pre>
          </div>
          <CopyToClipboard
            text={regeneratedKey}
            onCopy={() => NotificationManager.success("Virtual Key copied to clipboard")}
          >
            <Button type="primary" style={{ marginTop: 12 }}>Copy Virtual Key</Button>
          </CopyToClipboard>
        </div>
      ) : (
        <Form
          form={form}
          layout="vertical"
          onValuesChange={(changedValues) => {
            if ("duration" in changedValues) {
              setRegenerateFormData((prev: { duration?: string }) => ({ ...prev, duration: changedValues.duration }));
            }
          }}
        >
          <Form.Item name="key_alias" label="Key Alias">
            <Input disabled={true} />
          </Form.Item>
          <Form.Item name="max_budget" label="Max Budget (USD)">
            <InputNumber step={0.01} precision={2} style={{ width: "100%" }} />
          </Form.Item>
          <Form.Item name="tpm_limit" label="TPM Limit">
            <InputNumber style={{ width: "100%" }} />
          </Form.Item>
          <Form.Item name="rpm_limit" label="RPM Limit">
            <InputNumber style={{ width: "100%" }} />
          </Form.Item>
          <Form.Item name="duration" label="Expire Key (eg: 30s, 30h, 30d)" style={{ marginTop: 32 }}>
            <Input placeholder="" />
          </Form.Item>
          <Text type="secondary" style={{ fontSize: 12 }}>
            Current expiry: {selectedToken?.expires ? new Date(selectedToken.expires).toLocaleString() : "Never"}
          </Text>
          {newExpiryTime && (
            <Text type="success" style={{ marginTop: 8, fontSize: 12, display: "block" }}>New expiry: {newExpiryTime}</Text>
          )}
          <Form.Item
            name="grace_period"
            label="Grace Period (eg: 24h, 2d)"
            tooltip="Keep the old key valid for this duration after rotation. Both keys work during this period for seamless cutover. Empty = immediate revoke."
            style={{ marginTop: 32 }}
            rules={[
              {
                pattern: /^(\d+(s|m|h|d|w|mo))?$/,
                message: "Must be a duration like 30s, 30m, 24h, 2d, 1w, or 1mo",
              },
            ]}
          >
            <Input placeholder="e.g. 24h, 2d (empty = immediate revoke)" />
          </Form.Item>
          <Text type="secondary" style={{ fontSize: 12 }}>
            Recommended: 24h to 72h for production keys to allow seamless client migration.
          </Text>
        </Form>
      )}
    </Modal>
  );
}
