import { useCloudZeroUpdateSettings } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Form, Input, message, Modal } from "antd";
import { useEffect } from "react";
import { CloudZeroSettings } from "./types";

interface CloudZeroUpdateModalProps {
  open: boolean;
  onOk: () => void;
  onCancel: () => void;
  settings: CloudZeroSettings;
}

export default function CloudZeroUpdateModal({ open, onOk, onCancel, settings }: CloudZeroUpdateModalProps) {
  const { accessToken } = useAuthorized();
  const [form] = Form.useForm();
  const updateMutation = useCloudZeroUpdateSettings(accessToken || "");

  useEffect(() => {
    if (open && settings) {
      form.setFieldsValue({
        connection_id: settings.connection_id,
        timezone: settings.timezone || "UTC",
        api_key: "",
      });
    } else if (open) {
      form.resetFields();
    }
  }, [open, settings, form]);

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      updateMutation.mutate(
        {
          connection_id: values.connection_id,
          timezone: values.timezone || "UTC",
          ...(values.api_key && { api_key: values.api_key }),
        },
        {
          onSuccess: () => {
            message.success("CloudZero integration updated successfully");
            form.resetFields();
            onOk();
          },
          onError: (error: any) => {
            if (error?.errorFields) {
              return;
            }
            message.error(error?.message || "Failed to update CloudZero integration");
          },
        },
      );
    } catch (error: any) {
      if (error?.errorFields) {
        return;
      }
      message.error(error?.message || "Failed to update CloudZero integration");
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onCancel();
  };

  return (
    <Modal
      title="Edit CloudZero Integration"
      open={open}
      onOk={handleSubmit}
      onCancel={handleCancel}
      confirmLoading={updateMutation.isPending}
      okText={updateMutation.isPending ? "Updating..." : "Update"}
      cancelText="Cancel"
      okButtonProps={{
        disabled: updateMutation.isPending,
      }}
      cancelButtonProps={{
        disabled: updateMutation.isPending,
      }}
    >
      <Form form={form} layout="vertical" onFinish={handleSubmit}>
        <Form.Item
          label="CloudZero API Key"
          name="api_key"
          rules={[{ required: false, message: "Please enter your CloudZero API key" }]}
          tooltip="Leave empty to keep the existing API key"
        >
          <Input.Password placeholder="Leave empty to keep existing" />
        </Form.Item>
        <Form.Item
          label="Connection ID"
          name="connection_id"
          rules={[{ required: true, message: "Please enter your CloudZero connection ID" }]}
        >
          <Input placeholder="Enter your CloudZero connection ID" />
        </Form.Item>
        <Form.Item
          label="Timezone"
          name="timezone"
          tooltip="Timezone for date handling (defaults to UTC if not provided)"
        >
          <Input placeholder="UTC" />
        </Form.Item>
      </Form>
    </Modal>
  );
}
