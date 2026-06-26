import { useCloudZeroUpdateSettings } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroSettings";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { Form, Input, Modal } from "antd";
import MessageManager from "@/components/molecules/message_manager";
import { useEffect } from "react";
import { CloudZeroSettings } from "./types";
import { useTranslation } from "react-i18next";

interface CloudZeroUpdateModalProps {
  open: boolean;
  onOk: () => void;
  onCancel: () => void;
  settings: CloudZeroSettings;
}

export default function CloudZeroUpdateModal({ open, onOk, onCancel, settings }: CloudZeroUpdateModalProps) {
  const { t } = useTranslation();
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
            MessageManager.success(t("cloudZero.cloudZeroUpdateModal.updateSuccess"));
            form.resetFields();
            onOk();
          },
          onError: (error: any) => {
            if (error?.errorFields) {
              return;
            }
            MessageManager.error(error?.message || t("cloudZero.cloudZeroUpdateModal.updateFailed"));
          },
        },
      );
    } catch (error: any) {
      if (error?.errorFields) {
        return;
      }
      MessageManager.error(error?.message || t("cloudZero.cloudZeroUpdateModal.updateFailed"));
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onCancel();
  };

  return (
    <Modal
      title={t("cloudZero.cloudZeroUpdateModal.title")}
      open={open}
      onOk={handleSubmit}
      onCancel={handleCancel}
      confirmLoading={updateMutation.isPending}
      okText={updateMutation.isPending ? t("cloudZero.cloudZeroUpdateModal.updating") : t("common.update")}
      cancelText={t("common.cancel")}
      okButtonProps={{
        disabled: updateMutation.isPending,
      }}
      cancelButtonProps={{
        disabled: updateMutation.isPending,
      }}
    >
      <Form form={form} layout="vertical" onFinish={handleSubmit}>
        <Form.Item
          label={t("cloudZero.cloudZeroUpdateModal.apiKeyLabel")}
          name="api_key"
          rules={[{ required: false, message: t("cloudZero.cloudZeroUpdateModal.apiKeyRuleMessage") }]}
          tooltip={t("cloudZero.cloudZeroUpdateModal.apiKeyTooltip")}
        >
          <Input.Password placeholder={t("cloudZero.cloudZeroUpdateModal.apiKeyPlaceholder")} />
        </Form.Item>
        <Form.Item
          label={t("cloudZero.cloudZeroUpdateModal.connectionIdLabel")}
          name="connection_id"
          rules={[{ required: true, message: t("cloudZero.cloudZeroUpdateModal.connectionIdRequired") }]}
        >
          <Input placeholder={t("cloudZero.cloudZeroUpdateModal.connectionIdPlaceholder")} />
        </Form.Item>
        <Form.Item
          label={t("cloudZero.cloudZeroUpdateModal.timezoneLabel")}
          name="timezone"
          tooltip={t("cloudZero.cloudZeroUpdateModal.timezoneTooltip")}
        >
          <Input placeholder="UTC" />
        </Form.Item>
      </Form>
    </Modal>
  );
}
