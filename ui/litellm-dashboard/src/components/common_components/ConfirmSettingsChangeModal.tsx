import React from "react";
import { Alert, Modal, Typography } from "antd";
import {
  MinusCircleOutlined,
  PlusCircleOutlined,
  EditOutlined,
} from "@ant-design/icons";

export interface SettingsChange {
  field: string;
  type: "removed" | "added" | "changed";
  details: string;
}

interface ConfirmSettingsChangeModalProps {
  isOpen: boolean;
  changes: SettingsChange[];
  onConfirm: () => void;
  onCancel: () => void;
  confirmLoading: boolean;
}

const CHANGE_CONFIG: Record<
  SettingsChange["type"],
  { color: string; icon: React.ReactNode; label: string }
> = {
  removed: {
    color: "#cf1322",
    icon: <MinusCircleOutlined style={{ color: "#cf1322" }} />,
    label: "Removed",
  },
  added: {
    color: "#389e0d",
    icon: <PlusCircleOutlined style={{ color: "#389e0d" }} />,
    label: "Added",
  },
  changed: {
    color: "#d48806",
    icon: <EditOutlined style={{ color: "#d48806" }} />,
    label: "Changed",
  },
};

export default function ConfirmSettingsChangeModal({
  isOpen,
  changes,
  onConfirm,
  onCancel,
  confirmLoading,
}: ConfirmSettingsChangeModalProps) {
  const { Text } = Typography;

  return (
    <Modal
      title="Review Changes"
      open={isOpen}
      onOk={onConfirm}
      onCancel={onCancel}
      confirmLoading={confirmLoading}
      okText={confirmLoading ? "Saving..." : "Confirm Changes"}
      cancelText="Cancel"
      okButtonProps={{ disabled: confirmLoading }}
      cancelButtonProps={{ disabled: confirmLoading }}
    >
      <div className="space-y-4">
        <Alert
          message="These defaults apply to all future users created via SSO, API, or SCIM."
          type="warning"
          showIcon
        />

        <div className="mt-4 space-y-2">
          {changes.map((change, index) => {
            const config = CHANGE_CONFIG[change.type];
            return (
              <div
                key={index}
                className="flex items-start gap-2 p-2 rounded"
                style={{ backgroundColor: `${config.color}08` }}
              >
                <span className="mt-0.5 flex-shrink-0">{config.icon}</span>
                <div>
                  <Text strong style={{ color: config.color }}>
                    {change.field}
                  </Text>
                  <Text className="ml-1" style={{ color: config.color }}>
                    — {change.details}
                  </Text>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </Modal>
  );
}
