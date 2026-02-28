import { useEffect } from "react";
import { Modal, Form, Button, Typography, message } from "antd";
import { SaveOutlined } from "@ant-design/icons";
import { ProjectResponse } from "@/app/(dashboard)/hooks/projects/useProjects";
import {
  useUpdateProject,
  ProjectUpdateParams,
} from "@/app/(dashboard)/hooks/projects/useUpdateProject";
import { ProjectBaseForm, ProjectFormValues } from "./ProjectBaseForm";
import { buildProjectApiParams } from "./projectFormUtils";

interface EditProjectModalProps {
  isOpen: boolean;
  project: ProjectResponse;
  onClose: () => void;
  onSuccess?: () => void;
}

export function EditProjectModal({
  isOpen,
  project,
  onClose,
  onSuccess,
}: EditProjectModalProps) {
  const [form] = Form.useForm<ProjectFormValues>();
  const updateMutation = useUpdateProject();

  // Populate form with existing project data when modal opens
  useEffect(() => {
    if (isOpen && project) {
      // Model limits are stored inside metadata by the backend
      const metadataObj = (project.metadata ?? {}) as Record<string, unknown>;
      const rpmLimits = (metadataObj.model_rpm_limit ?? {}) as Record<string, number>;
      const tpmLimits = (metadataObj.model_tpm_limit ?? {}) as Record<string, number>;

      const modelLimits: ProjectFormValues["modelLimits"] = [];
      const allLimitModels = new Set([
        ...Object.keys(rpmLimits),
        ...Object.keys(tpmLimits),
      ]);
      for (const model of allLimitModels) {
        modelLimits.push({
          model,
          rpm: rpmLimits[model],
          tpm: tpmLimits[model],
        });
      }

      // Filter out internal keys from user-facing metadata
      const internalKeys = new Set(["model_rpm_limit", "model_tpm_limit"]);
      const metadata: ProjectFormValues["metadata"] = [];
      for (const [key, value] of Object.entries(metadataObj)) {
        if (!internalKeys.has(key)) {
          metadata.push({ key, value: String(value) });
        }
      }

      form.setFieldsValue({
        project_alias: project.project_alias ?? "",
        team_id: project.team_id ?? "",
        description: project.description ?? "",
        models: project.models ?? [],
        max_budget: project.litellm_budget_table?.max_budget ?? undefined,
        isBlocked: project.blocked,
        modelLimits: modelLimits.length > 0 ? modelLimits : undefined,
        metadata: metadata.length > 0 ? metadata : undefined,
      });
    }
  }, [isOpen, project, form]);

  const handleSubmit = async () => {
    try {
      const values = await form.validateFields();
      const params: ProjectUpdateParams = {
        ...buildProjectApiParams(values),
        team_id: values.team_id,
      };

      updateMutation.mutate(
        { projectId: project.project_id, params },
        {
          onSuccess: () => {
            message.success("Project updated successfully");
            onSuccess?.();
            onClose();
          },
          onError: (error) => {
            message.error(error.message || "Failed to update project");
          },
        },
      );
    } catch (error) {
      console.error("Validation failed:", error);
    }
  };

  return (
    <Modal
      title={
        <Typography.Text strong style={{ fontSize: 18 }}>
          Edit Project
        </Typography.Text>
      }
      open={isOpen}
      onCancel={onClose}
      width={720}
      destroyOnHidden
      footer={[
        <Button key="cancel" onClick={onClose}>
          Cancel
        </Button>,
        <Button
          key="submit"
          type="primary"
          icon={<SaveOutlined />}
          loading={updateMutation.isPending}
          onClick={handleSubmit}
        >
          Save Changes
        </Button>,
      ]}
    >
      <ProjectBaseForm form={form} />
    </Modal>
  );
}
