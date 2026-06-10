import { Form, Modal, Input } from "antd";
import MessageManager from "@/components/molecules/message_manager";
import { useEffect } from "react";
import useAuthorized from "@/app/(dashboard)/hooks/useAuthorized";
import { useCloudZeroCreate } from "@/app/(dashboard)/hooks/cloudzero/useCloudZeroCreate";
import { useTranslation } from "react-i18next";

interface CloudZeroCreationModalProps {
  open: boolean;
  onOk: () => void;
  onCancel: () => void;
}

export default function CloudZeroCreationModal({ open, onOk, onCancel }: CloudZeroCreationModalProps) {
  const { t } = useTranslation();
  const { accessToken } = useAuthorized();
  const [form] = Form.useForm();
  const createMutation = useCloudZeroCreate(accessToken || "");

  useEffect(() => {
    if (open) {
      form.resetFields();
    }
  }, [open, form]);

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      createMutation.mutate(
        {
          connection_id: values.connection_id,
          timezone: values.timezone || "UTC",
          ...(values.api_key && { api_key: values.api_key }),
        },
        {
          onSuccess: () => {
            MessageManager.success(t("cloudZero.cloudZeroCreateModal.createSuccess"));
            form.resetFields();
            onOk();
          },
          onError: (error: any) => {
            if (error?.errorFields) {
              return;
            }
            MessageManager.error(error?.message || t("cloudZero.cloudZeroCreateModal.createFailed"));
          },
        },
      );
    } catch (error: any) {
      if (error?.errorFields) {
        return;
      }
      MessageManager.error(error?.message || t("cloudZero.cloudZeroCreateModal.createFailed"));
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onCancel();
  };

  return (
    <Modal
      title={t("cloudZero.cloudZeroCreateModal.title")}
      open={open}
      onOk={handleSubmit}
      onCancel={handleCancel}
      confirmLoading={createMutation.isPending}
      okText={createMutation.isPending ? t("cloudZero.cloudZeroCreateModal.creating") : t("common.create")}
      cancelText={t("common.cancel")}
      okButtonProps={{
        disabled: createMutation.isPending,
      }}
      cancelButtonProps={{
        disabled: createMutation.isPending,
      }}
    >
      <Form form={form} layout="vertical" onFinish={handleSubmit}>
        <Form.Item
          label={t("cloudZero.cloudZeroCreateModal.apiKeyLabel")}
          name="api_key"
          rules={[{ required: true, message: t("cloudZero.cloudZeroCreateModal.apiKeyRequired") }]}
        >
          <Input.Password placeholder={t("cloudZero.cloudZeroCreateModal.apiKeyPlaceholder")} />
        </Form.Item>
        <Form.Item
          label={t("cloudZero.cloudZeroCreateModal.connectionIdLabel")}
          name="connection_id"
          rules={[{ required: true, message: t("cloudZero.cloudZeroCreateModal.connectionIdRequired") }]}
        >
          <Input placeholder={t("cloudZero.cloudZeroCreateModal.connectionIdPlaceholder")} />
        </Form.Item>
        <Form.Item
          label={t("cloudZero.cloudZeroCreateModal.timezoneLabel")}
          name="timezone"
          tooltip={t("cloudZero.cloudZeroCreateModal.timezoneTooltip")}
        >
          <Input placeholder="UTC" />
        </Form.Item>
      </Form>
    </Modal>
  );
}
