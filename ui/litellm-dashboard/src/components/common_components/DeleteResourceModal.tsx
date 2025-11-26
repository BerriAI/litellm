import { Alert, Descriptions, Input, Modal, Typography } from "antd";
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
        <div className="mt-4 p-4 bg-red-50 rounded-lg border border-red-200">
          <Title level={5} className="mb-3 text-gray-900">
            {resourceInformationTitle}
          </Title>
          <Descriptions column={1} size="small">
            {resourceInformation &&
              resourceInformation.map(({ label, value, ...textProps }) => (
                <Descriptions.Item key={label} label={<span className="font-semibold text-gray-700">{label}</span>}>
                  <Text {...textProps}>{value ?? "-"}</Text>
                </Descriptions.Item>
              ))}
          </Descriptions>
        </div>
        <div>
          <Text>{message}</Text>
        </div>
        {requiredConfirmation && (
          <div className="mb-6 mt-4 pt-4 border-t border-gray-200">
            <Text className="block text-base font-medium text-gray-700 mb-2">
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
              className="rounded-md text-base border-gray-200"
              autoFocus
            />
          </div>
        )}
      </div>
    </Modal>
  );
}
