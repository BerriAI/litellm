"use client";

import { useState } from "react";
import { Modal } from "antd";
import { FolderPlus } from "lucide-react";

import { toast } from "@/lib/toast";
import { useZodForm } from "@/lib/forms/useZodForm";
import { Button } from "@/components/ui/button";
import { UiLoadingSpinner } from "@/components/ui/ui-loading-spinner";
import { useCreateProject, ProjectCreateParams } from "@/app/(dashboard)/hooks/projects/useCreateProject";
import { ProjectBaseForm } from "./ProjectBaseForm";
import { emptyProjectFormValues, projectFormSchema } from "./projectFormSchema";
import { buildProjectApiParams } from "./projectFormUtils";

interface CreateProjectModalProps {
  isOpen: boolean;
  onClose: () => void;
}

function CreateProjectForm({ onClose }: { onClose: () => void }) {
  const form = useZodForm(projectFormSchema, { defaultValues: emptyProjectFormValues });
  const createMutation = useCreateProject();
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const handleSubmit = form.handleSubmit((values) => {
    const params: ProjectCreateParams = {
      ...buildProjectApiParams(values),
      team_id: values.team_id,
    };

    createMutation.mutate(params, {
      onSuccess: () => {
        toast.success("Project created successfully");
        form.reset(emptyProjectFormValues);
        onClose();
      },
      onError: (error) => {
        toast.error(error.message || "Failed to create project");
      },
    });
  });

  const handleCancel = () => {
    form.reset(emptyProjectFormValues);
    onClose();
  };

  return (
    <form onSubmit={(event) => event.preventDefault()}>
      <ProjectBaseForm form={form} advancedOpen={advancedOpen} onAdvancedOpenChange={setAdvancedOpen} />

      <div className="mt-6 flex justify-end gap-2 border-t border-border pt-4">
        <Button type="button" variant="outline" onClick={handleCancel}>
          Cancel
        </Button>
        <Button type="button" onClick={() => void handleSubmit()} disabled={createMutation.isPending}>
          {createMutation.isPending ? <UiLoadingSpinner /> : <FolderPlus />}
          Create Project
        </Button>
      </div>
    </form>
  );
}

export function CreateProjectModal({ isOpen, onClose }: CreateProjectModalProps) {
  return (
    <Modal
      title={<span className="text-lg font-semibold text-foreground">Create New Project</span>}
      open={isOpen}
      onCancel={onClose}
      width={720}
      destroyOnHidden
      footer={null}
    >
      <CreateProjectForm onClose={onClose} />
    </Modal>
  );
}
