import React, { useEffect, useState } from "react";
import { Button, Text, TextInput, Title, Grid, Col } from "@tremor/react";
import { Modal, Form, InputNumber } from "antd";
import { add } from "date-fns";
import { regenerateKeyCall } from "../networking";
import { KeyResponse } from "../key_team_helpers/key_list";
import { CopyToClipboard } from "react-copy-to-clipboard";
import NotificationManager from "../molecules/notifications_manager";

interface RegenerateKeyModalProps {
  selectedToken: KeyResponse | null;
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  premiumUser: boolean;
  setAccessToken?: (token: string) => void;
  onKeyUpdate?: (updatedKeyData: Partial<KeyResponse>) => void;
}

export function RegenerateKeyModal({
  selectedToken,
  visible,
  onClose,
  accessToken,
  premiumUser,
  setAccessToken,
  onKeyUpdate,
}: RegenerateKeyModalProps) {
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
      NotificationManager.success("API Key regenerated successfully");

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

      // If user regenerated their own auth key, update both local and global access tokens
      if (isOwnKey) {
        setCurrentAccessToken(response.key); // Update local token immediately
        if (setAccessToken) {
          setAccessToken(response.key); // Update global token
        }
      }

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
      title="Regenerate API Key"
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
              <Button key="cancel" onClick={handleClose} className="mr-2">
                Cancel
              </Button>,
              <Button key="regenerate" onClick={handleRegenerateKey} disabled={isRegenerating}>
                {isRegenerating ? "Regenerating..." : "Regenerate"}
              </Button>,
            ]
      }
    >
      {regeneratedKey ? (
        <Grid numItems={1} className="gap-2 w-full">
          <Title>Regenerated Key</Title>
          <Col numColSpan={1}>
            <p>
              Please replace your old key with the new key generated. For security reasons,{" "}
              <b>you will not be able to view it again</b> through your LiteLLM account. If you lose this secret key,
              you will need to generate a new one.
            </p>
          </Col>
          <Col numColSpan={1}>
            <Text className="mt-3">Key Alias:</Text>
            <div className="bg-gray-100 p-2 rounded mb-2">
              <pre className="break-words whitespace-normal">{selectedToken?.key_alias || "No alias set"}</pre>
            </div>
            <Text className="mt-3">New API Key:</Text>
            <div className="bg-gray-100 p-2 rounded mb-2">
              <pre className="break-words whitespace-normal">{regeneratedKey}</pre>
            </div>
            <CopyToClipboard
              text={regeneratedKey}
              onCopy={() => NotificationManager.success("API Key copied to clipboard")}
            >
              <Button className="mt-3">Copy API Key</Button>
            </CopyToClipboard>
          </Col>
        </Grid>
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
            <TextInput disabled={true} />
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
          <Form.Item name="duration" label="Expire Key (eg: 30s, 30h, 30d)" className="mt-8">
            <TextInput placeholder="" />
          </Form.Item>
          <div className="mt-2 text-sm text-gray-500">
            Current expiry: {selectedToken?.expires ? new Date(selectedToken.expires).toLocaleString() : "Never"}
          </div>
          {newExpiryTime && <div className="mt-2 text-sm text-green-600">New expiry: {newExpiryTime}</div>}
        </Form>
      )}
    </Modal>
  );
}
