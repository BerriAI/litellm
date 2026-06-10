import { Modal, Form, Button, Typography } from "antd";
import { FolderAddOutlined } from "@ant-design/icons";
import MessageManager from "@/components/molecules/message_manager";
import { useCreateProject, ProjectCreateParams } from "@/app/(dashboard)/hooks/projects/useCreateProject";
import { useTranslation } from "react-i18next";
import { ProjectBaseForm, ProjectFormValues } from "./ProjectBaseForm";
import { buildProjectApiParams } from "./projectFormUtils";

interface CreateProjectModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function CreateProjectModal({ isOpen, onClose }: CreateProjectModalProps) {
  const { t } = useTranslation();
  const [form] = Form.useForm<ProjectFormValues>();
  const createMutation = useCreateProject();

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      const params: ProjectCreateParams = {
        ...buildProjectApiParams(values),
        team_id: values.team_id,
      };

      createMutation.mutate(params, {
        onSuccess: () => {
          MessageManager.success(t("projects.createProjectModal.createSuccess"));
          form.resetFields();
          onClose();
        },
        onError: (error) => {
          MessageManager.error(error.message || t("projects.createProjectModal.createFailed"));
        },
      });
    } catch (error) {
      console.error("Validation failed:", error);
    }
  };

  const handleCancel = () => {
    form.resetFields();
    onClose();
  };

  return (
    <Modal
      title={
        <Typography.Text strong style={{ fontSize: 18 }}>
          {t("projects.createProjectModal.title")}
        </Typography.Text>
      }
      open={isOpen}
      onCancel={handleCancel}
      width={720}
      destroyOnHidden
      footer={[
        <Button key="cancel" onClick={handleCancel}>
          {t("common.cancel")}
        </Button>,
        <Button
          key="submit"
          type="primary"
          icon={<FolderAddOutlined />}
          loading={createMutation.isPending}
          onClick={handleSubmit}
        >
          {t("projects.createProjectModal.createBtn")}
        </Button>,
      ]}
    >
      <ProjectBaseForm form={form} />
    </Modal>
  );
}
