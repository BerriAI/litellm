"use client";

import { StoreRequestInSpendLogsParams, useStoreRequestInSpendLogs } from "@/app/(dashboard)/hooks/storeRequestInSpendLogs/useStoreRequestInSpendLogs";
import NotificationsManager from "@/components/molecules/notifications_manager";
import { parseErrorMessage } from "@/components/shared/errorUtils";
import { ClockCircleOutlined } from "@ant-design/icons";
import { Button, Form, Input, Modal, Space, Switch } from "antd";
import React from "react";

interface SpendLogsSettingsModalProps {
  isVisible: boolean;
  onCancel: () => void;
  onSuccess?: () => void;
}

const SpendLogsSettingsModal: React.FC<SpendLogsSettingsModalProps> = ({ isVisible, onCancel, onSuccess }) => {
  const [form] = Form.useForm();
  const { mutateAsync, isPending } = useStoreRequestInSpendLogs();
  const storePromptsValue = Form.useWatch('store_prompts_in_spend_logs', form);

  const handleFormSubmit = async (formValues: StoreRequestInSpendLogsParams) => {
    try {
      await mutateAsync(formValues, {
        onSuccess: () => {
          NotificationsManager.success("Spend logs settings updated successfully");
          form.resetFields();
          onSuccess?.();
        },
        onError: (error) => {
          NotificationsManager.fromBackend("Failed to save spend logs settings: " + parseErrorMessage(error));
        },
      });
    } catch (error) {
      NotificationsManager.fromBackend("Failed to save spend logs settings: " + parseErrorMessage(error));
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onCancel();
  };

  return (
    <Modal
      title="Spend Logs Settings"
      open={isVisible}
      width={600}
      footer={
        <Space>
          <Button onClick={handleCancel} disabled={isPending}>
            Cancel
          </Button>
          <Button type="primary" loading={isPending} onClick={() => form.submit()}>
            {isPending ? "Saving..." : "Save Settings"}
          </Button>
        </Space>
      }
      onCancel={handleCancel}
    >
      <Form
        form={form}
        layout="horizontal"
        labelCol={{ flex: "auto", style: { textAlign: "left" } }}
        wrapperCol={{ flex: "auto", style: { textAlign: "right" } }}
        onFinish={handleFormSubmit}
        initialValues={{
          store_prompts_in_spend_logs: false,
          maximum_spend_logs_retention_period: undefined,
        }}
      >
        <Form.Item
          name="store_prompts_in_spend_logs"
          tooltip="When enabled, prompts will be stored in spend logs for tracking and analysis purposes."
          valuePropName="checked"
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <span>Store Prompts in Spend Logs</span>
            <Switch checked={storePromptsValue ?? false} onChange={(checked) => form.setFieldValue('store_prompts_in_spend_logs', checked)} />
          </div>
        </Form.Item>


        <Form.Item
          label="Maximum Spend Logs Retention Period (Optional)"
          name="maximum_spend_logs_retention_period"
          tooltip="Set the maximum retention period for spend logs (e.g., '7d' for 7 days, '30d' for 30 days). Leave empty for no limit."
          labelCol={{ flex: "auto", style: { textAlign: "left" } }}
          wrapperCol={{ flex: "0 0 25%", style: { textAlign: "right" } }}
        >
          <Input
            placeholder="e.g., 7d, 30d"
            prefix={<ClockCircleOutlined />}
          />
        </Form.Item>
      </Form>
    </Modal>
  );
};

export default SpendLogsSettingsModal;
