import { FolderPlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Form } from "antd";
import MessageManager from "@/components/molecules/message_manager";
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
          MessageManager.success("Project created successfully");
          form.resetFields();
          onClose();
        },
        onError: (error) => {
          MessageManager.error(error.message || "Failed to create project");
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
    <Dialog
      open={isOpen}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent className="max-w-[720px]">
        <DialogHeader>
          <DialogTitle className="text-lg font-semibold">
            Create New Project
          </DialogTitle>
        </DialogHeader>
        <ProjectBaseForm form={form} />
        <DialogFooter>
          <Button variant="outline" onClick={handleCancel}>
            Cancel
          </Button>
          <Button
            onClick={handleSubmit}
            disabled={createMutation.isPending}
          >
            <FolderPlus className="h-4 w-4" />
            {createMutation.isPending ? "Creating…" : "Create Project"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
