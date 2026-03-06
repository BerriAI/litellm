import { Alert, Card, Descriptions, Input, Modal, Typography, theme } from "antd";
import { ExclamationCircleOutlined } from "@ant-design/icons";
import React, { useState, useEffect } from "react";

interface DeleteResourceModalProps {
  isOpen: boolean;
  title: string;
  alertMessage?: string;
  message: string;
  resourceInformationTitle?: string;
  resourceInformation?: Array<
    {
      label: string;
      value: string | number | undefined | null;
    } & Omit<React.ComponentProps<typeof Typography.Text>, "children">
  >;
  onCancel: () => void;
  onOk: () => void;
  confirmLoading: boolean;
  requiredConfirmation?: string;
}

export default function DeleteResourceModal({
  isOpen,
  title,
  alertMessage,
  message,
  resourceInformationTitle,
  resourceInformation,
  onCancel,
  onOk,
  confirmLoading,
  requiredConfirmation,
}: DeleteResourceModalProps) {
  const { Title, Text } = Typography;
  const { token } = theme.useToken();
  const [requiredConfirmationInput, setRequiredConfirmationInput] = useState("");

  useEffect(() => {
    if (isOpen) {
      setRequiredConfirmationInput("");
    }
  }, [isOpen]);

  return (
    <Modal
      title={title}
      open={isOpen}
      onOk={onOk}
      onCancel={onCancel}
      confirmLoading={confirmLoading}
      okText={confirmLoading ? "Deleting..." : "Delete"}
      cancelText="Cancel"
      okButtonProps={{
        danger: true,
        disabled: (!!requiredConfirmation && requiredConfirmationInput !== requiredConfirmation) || confirmLoading,
      }}
      cancelButtonProps={{ disabled: confirmLoading }}
    >
      <div className="space-y-4">
        {alertMessage && <Alert message={alertMessage} type="warning" />}
        <Card
          title={resourceInformationTitle}
          className="mt-4"
          styles={{
            body: { padding: "16px" },
            header: {
              backgroundColor: token.colorErrorBg,
              borderColor: token.colorErrorBorder,
            },
          }}
          style={{
            backgroundColor: token.colorErrorBg,
            borderColor: token.colorErrorBorder,
          }}
        >
          <Descriptions column={1} size="small">
            {resourceInformation &&
              resourceInformation.map(({ label, value, ...textProps }) => (
                <Descriptions.Item key={label} label={<span className="font-semibold">{label}</span>}>
                  <Text {...textProps}>{value ?? "-"}</Text>
                </Descriptions.Item>
              ))}
          </Descriptions>
        </Card>
        <div>
          <Text>{message}</Text>
        </div>
        {requiredConfirmation && (
          <div className="mb-6 mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
            <Text className="block text-base font-medium text-gray-700 dark:text-gray-300 mb-2">
              <Text>Type </Text>
              <Text strong type="danger">
                {requiredConfirmation}
              </Text>
              <Text> to confirm deletion:</Text>
            </Text>
            <Input
              value={requiredConfirmationInput}
              onChange={(e) => setRequiredConfirmationInput(e.target.value)}
              placeholder={requiredConfirmation}
              className="rounded-md"
              prefix={<ExclamationCircleOutlined style={{ color: token.colorError }} />}
              autoFocus
            />
          </div>
        )}
      </div>
    </Modal>
  );
}
