import { Modal, Form, Button, Typography, message } from "antd";
import { FolderAddOutlined } from "@ant-design/icons";
import {
  useCreateProject,
  ProjectCreateParams,
} from "@/app/(dashboard)/hooks/projects/useCreateProject";
import {
  ProjectBaseForm,
  ProjectFormValues,
} from "./ProjectBaseForm";
import { buildProjectApiParams } from "./projectFormUtils";

interface CreateProjectModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function CreateProjectModal({
  isOpen,
  onClose,
}: CreateProjectModalProps) {
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
          message.success("Project created successfully");
          form.resetFields();
          onClose();
        },
        onError: (error) => {
          message.error(error.message || "Failed to create project");
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
          Create New Project
        </Typography.Text>
      }
      open={isOpen}
      onCancel={handleCancel}
      width={720}
      destroyOnHidden
      footer={[
        <Button key="cancel" onClick={handleCancel}>
          Cancel
        </Button>,
        <Button
          key="submit"
          type="primary"
          icon={<FolderAddOutlined />}
          loading={createMutation.isPending}
          onClick={handleSubmit}
        >
          Create Project
        </Button>,
      ]}
    >
      <ProjectBaseForm form={form} />
    </Modal>
  );
}
