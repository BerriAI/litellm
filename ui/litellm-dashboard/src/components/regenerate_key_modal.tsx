import React, { useEffect, useState } from "react";
import { Button, Text, TextInput, Title, Grid, Col } from "@tremor/react";
import { Modal, Form, InputNumber, message } from "antd";
import { add } from "date-fns";
import { regenerateKeyCall } from "./networking";
import { KeyResponse } from "./key_team_helpers/key_list";
import { CopyToClipboard } from "react-copy-to-clipboard";

interface RegenerateKeyModalProps {
  selectedToken: KeyResponse | null;
  visible: boolean;
  onClose: () => void;
  accessToken: string | null;
  premiumUser: boolean;
}

export function RegenerateKeyModal({
  selectedToken,
  visible,
  onClose,
  accessToken,
  premiumUser,
}: RegenerateKeyModalProps) {
  const [form] = Form.useForm();
  const [regeneratedKey, setRegeneratedKey] = useState<string | null>(null);
  const [regenerateFormData, setRegenerateFormData] = useState<any>(null);
  const [newExpiryTime, setNewExpiryTime] = useState<string | null>(null);
  const [isRegenerating, setIsRegenerating] = useState(false);

  useEffect(() => {
    if (visible && selectedToken) {
      form.setFieldsValue({
        key_alias: selectedToken.key_alias,
        max_budget: selectedToken.max_budget,
        tpm_limit: selectedToken.tpm_limit,
        rpm_limit: selectedToken.rpm_limit,
        duration: selectedToken.duration || "",
      });
    }
  }, [visible, selectedToken, form]);

  useEffect(() => {
    if (!visible) {
      // Reset states when modal is closed
      setRegeneratedKey(null);
      setIsRegenerating(false);
      form.resetFields();
    }
  }, [visible, form]);

  useEffect(() => {
    const calculateNewExpiryTime = (duration: string | undefined) => {
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
      const response = await regenerateKeyCall(accessToken, selectedToken.token, formValues);
      setRegeneratedKey(response.key);
      message.success("API Key regenerated successfully");
    } catch (error) {
      console.error("Error regenerating key:", error);
      message.error("Failed to regenerate API Key");
      setIsRegenerating(false); // Reset regenerating state on error
    }
  };

  const handleClose = () => {
    setRegeneratedKey(null);
    setIsRegenerating(false);
    form.resetFields();
    onClose();
  };

  return (
    <Modal
      title="Regenerate API Key"
      open={visible}
      onCancel={handleClose}
      footer={regeneratedKey ? [
        <Button key="close" onClick={handleClose}>
          Close
        </Button>,
      ] : [
        <Button key="cancel" onClick={handleClose} className="mr-2">
          Cancel
        </Button>,
        <Button 
          key="regenerate" 
          onClick={handleRegenerateKey}
          disabled={isRegenerating}
        >
          {isRegenerating ? "Regenerating..." : "Regenerate"}
        </Button>,
      ]}
    >
      {regeneratedKey ? (
        <Grid numItems={1} className="gap-2 w-full">
          <Title>Regenerated Key</Title>
          <Col numColSpan={1}>
            <p>
              Please replace your old key with the new key generated. For
              security reasons, <b>you will not be able to view it again</b>{" "}
              through your LiteLLM account. If you lose this secret key, you
              will need to generate a new one.
            </p>
          </Col>
          <Col numColSpan={1}>
            <Text className="mt-3">Key Alias:</Text>
            <div className="bg-gray-100 p-2 rounded mb-2">
              <pre className="break-words whitespace-normal">
                {selectedToken?.key_alias || "No alias set"}
              </pre>
            </div>
            <Text className="mt-3">New API Key:</Text>
            <div className="bg-gray-100 p-2 rounded mb-2">
              <pre className="break-words whitespace-normal">{regeneratedKey}</pre>
            </div>
            <CopyToClipboard
              text={regeneratedKey}
              onCopy={() => message.success("API Key copied to clipboard")}
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
          {newExpiryTime && (
            <div className="mt-2 text-sm text-green-600">
              New expiry: {newExpiryTime}
            </div>
          )}
        </Form>
      )}
    </Modal>
  );
}