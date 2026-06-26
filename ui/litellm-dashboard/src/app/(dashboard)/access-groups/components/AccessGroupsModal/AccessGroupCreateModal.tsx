import React from "react";
import { Modal, Form } from "antd";
import { useTranslation } from "react-i18next";
import MessageManager from "@/components/molecules/message_manager";
import { AccessGroupBaseForm, AccessGroupFormValues } from "./AccessGroupBaseForm";
import {
  useCreateAccessGroup,
  AccessGroupCreateParams,
} from "@/app/(dashboard)/hooks/accessGroups/useCreateAccessGroup";

interface AccessGroupCreateModalProps {
  visible: boolean;
  onCancel: () => void;
  onSuccess?: () => void;
}

export function AccessGroupCreateModal({ visible, onCancel, onSuccess }: AccessGroupCreateModalProps) {
  const { t } = useTranslation();
  const [form] = Form.useForm<AccessGroupFormValues>();
  const createMutation = useCreateAccessGroup();

  const handleOk = () => {
    form
      .validateFields()
      .then((values) => {
        const params: AccessGroupCreateParams = {
          access_group_name: values.name,
          description: values.description,
          access_model_names: values.modelIds,
          access_mcp_server_ids: values.mcpServerIds,
          access_agent_ids: values.agentIds,
        };

        createMutation.mutate(params, {
          onSuccess: () => {
            MessageManager.success(t("accessGroups.accessGroupCreateModal.createSuccess"));
            form.resetFields();
            onSuccess?.();
            onCancel();
          },
        });
      })
      .catch((info) => {
        console.log("Validate Failed:", info);
      });
  };

  return (
    <Modal
      title={t("accessGroups.accessGroupCreateModal.title")}
      open={visible}
      onOk={handleOk}
      onCancel={onCancel}
      width={700}
      okText={t("accessGroups.accessGroupCreateModal.okText")}
      cancelText={t("common.cancel")}
      confirmLoading={createMutation.isPending}
      destroyOnClose
    >
      <AccessGroupBaseForm form={form} />
    </Modal>
  );
}
