import { FolderPlus } from "lucide-react";
import { useEffect } from "react";
import { FormProvider, useForm } from "react-hook-form";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import MessageManager from "@/components/molecules/message_manager";
import {
  useCreateProject,
  ProjectCreateParams,
} from "@/app/(dashboard)/hooks/projects/useCreateProject";
import {
  ProjectBaseForm,
  ProjectFormValues,
  emptyProjectFormValues,
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
  const form = useForm<ProjectFormValues>({
    defaultValues: emptyProjectFormValues,
    mode: "onSubmit",
  });
  const createMutation = useCreateProject();

  useEffect(() => {
    if (isOpen) form.reset(emptyProjectFormValues);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  const handleCancel = () => {
    form.reset(emptyProjectFormValues);
    onClose();
  };

  const onSubmit = form.handleSubmit((values) => {
    const params: ProjectCreateParams = {
      ...buildProjectApiParams(values),
      team_id: values.team_id,
    };
    createMutation.mutate(params, {
      onSuccess: () => {
        MessageManager.success("Project created successfully");
        form.reset(emptyProjectFormValues);
        onClose();
      },
      onError: (error) => {
        MessageManager.error(error.message || "Failed to create project");
      },
    });
  });

  return (
    <Dialog
      open={isOpen}
      onOpenChange={(o) => (!o ? handleCancel() : undefined)}
    >
      <DialogContent className="max-w-[720px] max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="text-lg font-semibold">
            Create New Project
          </DialogTitle>
          <DialogDescription className="sr-only">
            Create a new project within a team.
          </DialogDescription>
        </DialogHeader>
        <FormProvider {...form}>
          <form onSubmit={onSubmit}>
            <ProjectBaseForm />
            <DialogFooter className="mt-6">
              <Button
                type="button"
                variant="outline"
                onClick={handleCancel}
              >
                Cancel
              </Button>
              <Button type="submit" disabled={createMutation.isPending}>
                <FolderPlus className="h-4 w-4" />
                {createMutation.isPending ? "Creating…" : "Create Project"}
              </Button>
            </DialogFooter>
          </form>
        </FormProvider>
      </DialogContent>
    </Dialog>
  );
}
